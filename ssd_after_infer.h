//
// Created by czx on 2019/11/14.
//

#ifndef VEGAHISI_VEGA_SSD_AFTER_INFER_H
#define VEGAHISI_VEGA_SSD_AFTER_INFER_H

#include "graph/vega_base_infer_op.h"
#include <typeinfo>
namespace vega {

    class SsdConfig {
    public:
        SsdConfig() = default;

        virtual ~SsdConfig() = default;

        typedef struct {
            std::vector<float> min_size;
            std::vector<float> max_size;
            std::vector<float> aspect_ratio;
            bool flip;
            bool clip;
            std::vector<float> variance;
            int step_w;
            int step_h;
            float offset;
            std::string loc_layer;
            std::string conf_layer;
            int width;
            int height;
        } PriorBoxProperty;

        DgError load(const std::string &cfgPath, int &ClassNum, float &NmsThresh,
                     std::vector<PriorBoxProperty> &PriorBoxProperty_) {
            ClassNum = 0;
            NmsThresh = 0;
            SSDSplitType = 2;
            ConfCatLayer.resize(0);
            LocCatLayer.resize(0);
            JsonConfig config;
            config.setKeyIgnoreCase(true);
            if (!config.load(cfgPath)) {
                LOG(ERROR) << "Load file failed: " << cfgPath;
                return DG_ERR_NOT_EXIST;
            }

            get(config, "SSDSplitType", SSDSplitType);
            get(config, "ConfCatLayer", ConfCatLayer);
            get(config, "LocCatLayer", LocCatLayer);

            get(config, "ClassNum", ClassNum);
            get(config, "NmsThresh", NmsThresh);
            int anchor_num = 0;
            get(config, "PriorBox/Size", anchor_num);
            PriorBoxProperty_.resize(anchor_num);
            for (int i = 0; i < anchor_num; i++) {
                std::string prefix1 = "PriorBox" + std::to_string(i) + "/";

#define GET_VECTOR(key, v) \
                { \
                    int n=0; \
                    get(config,prefix1+key+"/Size",n); \
                    for(int j=0;j<n;j++) \
                    { \
                        std::string prefix2=key+std::to_string(j) ; \
                        float value; \
                        get(config,prefix1+prefix2,value);  \
                        v.push_back(value);  \
                    } \
                }
                GET_VECTOR("min_size", PriorBoxProperty_[i].min_size)
                GET_VECTOR("max_size", PriorBoxProperty_[i].max_size)
                GET_VECTOR("aspect_ratio", PriorBoxProperty_[i].aspect_ratio)
                get(config, prefix1 + "flip", PriorBoxProperty_[i].flip);
                get(config, prefix1 + "clip", PriorBoxProperty_[i].clip);
                GET_VECTOR("variance", PriorBoxProperty_[i].variance)
                get(config, prefix1 + "step_w", PriorBoxProperty_[i].step_w);
                get(config, prefix1 + "step_h", PriorBoxProperty_[i].step_h);
                get(config, prefix1 + "offset", PriorBoxProperty_[i].offset);
                get(config, prefix1 + "width", PriorBoxProperty_[i].width);
                get(config, prefix1 + "height", PriorBoxProperty_[i].height);
                get(config, prefix1 + "loc_layer", PriorBoxProperty_[i].loc_layer);
                get(config, prefix1 + "conf_layer", PriorBoxProperty_[i].conf_layer);
            }
            return DG_OK;
        }

        int getSplitType() {
            return SSDSplitType;
        }

        std::string getLocCatLayer() {
            return LocCatLayer;
        }

        std::string getConfCatLayer() {
            return ConfCatLayer;
        }

    private:
        int SSDSplitType;
        std::string LocCatLayer;
        std::string ConfCatLayer;

        template<typename T>
        DgError get(JsonConfig &config, const std::string &key, T &val) {
            auto &jv = config.Value(key);
            if (!jv.empty()) {
                val = static_cast<T>(jv);
            }
            return DG_OK;
        }
    };

    template<typename T>
    class VegaSsdAfterInfer {
    public:
        typedef void (*GetLayerDim)(const std::string &layer, std::vector<int> &Dim, void *op);

        VegaSsdAfterInfer(GetLayerDim F) { getEnginOutputLayerDim = F; };

        ~VegaSsdAfterInfer() = default;

    private:
        static const unsigned int COORDINATE_NUM = 4;
        typedef struct {
            int left;
            int top;
            int right;
            int bottom;
            bool is_del;
            float confidence;
            int type;
        } SsdBox;

        typedef struct {
            int x;
            int y;
            int prior_num;
            int type;
            float conf;
        }BoxInAnchor;


        std::vector<std::shared_ptr<int>> vPriorbox;
        T type_id;
        int OrigImgWidth;
        int OrigImgHeidth;
        int ClassNum;
        int SSDSplitType;
        std::string LocCatLayer;
        std::string ConfCatLayer;
        int EngineAlign = 32;
        float NmsThresh;
        std::vector<T> ConfThresh;
        std::vector<float> FConfThresh;
        std::vector<T> LogConfThresh;  //= log(thresh)
        std::vector<SsdConfig::PriorBoxProperty> PriorBoxProperty_;
        std::vector<int> perPriorBoxNum;
        std::vector<int> perPriorBoxSize; //equal perPriorBoxNum*priorbox_width*priorbox_height;
        int toTalPriorBoxSize; // the sum of all perPriorBoxSize;
        std::vector<int> prior_size_step; // step0=perPriorBoxSize[0] step1=step0+perPriorBoxSize[1].....
        GetLayerDim getEnginOutputLayerDim;
    public:
        void ssdInit(std::shared_ptr<ModelConfig> cfg, int engine_align, void *op,std::string ssd_config_path) {
            getProperties(cfg,ssd_config_path);
            OrigImgWidth = cfg->input_size_.width;
            OrigImgHeidth = cfg->input_size_.height;
            EngineAlign = engine_align;
            for (unsigned int i = 0; i < PriorBoxProperty_.size(); i++) {
                std::vector<int> Dim;
                (*getEnginOutputLayerDim)(PriorBoxProperty_[i].loc_layer, Dim, op);
                if (Dim.size() == 3) {
                    LOGFULL << "prior box  " << i << ": " << " chw: " << Dim[0] << " " << Dim[1] << " " << Dim[2];
                    PriorBoxProperty_[i].height = Dim[1];
                    PriorBoxProperty_[i].width = Dim[2];
                }
            }
            ConfThresh.resize(ClassNum);
            ConfThresh[0] = 1.0;
            LogConfThresh.resize(ClassNum);
            LogConfThresh[0] = 1.0;
            FConfThresh.resize(ClassNum);
            FConfThresh[0]=1.0;
            for (int i = 1; i < ClassNum; i++) {
                auto &def = cfg->tag_table_.getTagByIdx(i);
                FConfThresh[i]= def.threshold.end_ ;
                float log_conf=0;
                if(ClassNum==2)
                    log_conf=log(def.threshold.end_/(1.f-def.threshold.end_));
                else
                    log_conf=log(def.threshold.end_*ClassNum);
                if(typeid(type_id)==typeid(int)) {
                    ConfThresh[i] = (int) (def.threshold.end_ * 4096);
                    LogConfThresh[i] = (int) (log_conf * 4096);
                } else{
                    ConfThresh[i] = def.threshold.end_ ;
                    LogConfThresh[i] = log_conf;
                }
            }

            perPriorBoxNum.resize(PriorBoxProperty_.size());
            perPriorBoxSize.resize(PriorBoxProperty_.size());
            vPriorbox.resize(PriorBoxProperty_.size());
            toTalPriorBoxSize = 0;
            for (unsigned int i = 0; i < PriorBoxProperty_.size(); i++) {
                perPriorBoxNum[i] =PriorBoxProperty_[i].min_size.size() * (1 + PriorBoxProperty_[i].aspect_ratio.size()) +
                        PriorBoxProperty_[i].max_size.size();
                perPriorBoxSize[i] = PriorBoxProperty_[i].height * PriorBoxProperty_[i].width * perPriorBoxNum[i];
                int size = perPriorBoxSize[i] * COORDINATE_NUM;
                vPriorbox[i] = std::shared_ptr<int>(new int[size]);
                toTalPriorBoxSize += perPriorBoxSize[i];
            }
            for (unsigned int i = 0; i < PriorBoxProperty_.size(); i++) {
                priorBoxInit(vPriorbox[i].get(), PriorBoxProperty_[i], perPriorBoxNum[i]);
            }
            prior_size_step.resize(PriorBoxProperty_.size());
            prior_size_step[0] = perPriorBoxSize[0];
            for (unsigned int i = 1; i < PriorBoxProperty_.size(); i++) {
                prior_size_step[i] = prior_size_step[i - 1] + perPriorBoxSize[i];
            }
        }

        void ssdOutput(VegaBaseInferOpCtx *ctx, std::vector<float> &output, int batchIdx, void *op) {
            std::vector<std::vector<SsdBox>> vv_box;
            vv_box.resize(ClassNum);
            if (SSDSplitType == 1) {
                ssdCatOutput(ctx, batchIdx, vv_box, op);
            } else {
                std::vector<std::vector< BoxInAnchor>>  boxInAnchor;
                boxInAnchor.resize(PriorBoxProperty_.size());
                for (unsigned int i = 0; i < PriorBoxProperty_.size(); i++) {
                    DataBlob<T> conf = ctx->getBlob<T>(PriorBoxProperty_[i].conf_layer, batchIdx);
                    int intAlign = EngineAlign / sizeof(int);
                    int stride = (PriorBoxProperty_[i].width + intAlign - 1) / intAlign * intAlign;
                    if (ClassNum == 2) {
                        getConfScoreClassNum2(conf.begin(), perPriorBoxNum[i], PriorBoxProperty_[i].width, stride,PriorBoxProperty_[i].height,boxInAnchor[i]);
                    } else {
                        getConfScore(conf.begin(), perPriorBoxNum[i], PriorBoxProperty_[i].width, stride,PriorBoxProperty_[i].height, boxInAnchor[i]);
                    }
                    DataBlob<T> loc = ctx->getBlob<T>(PriorBoxProperty_[i].loc_layer, batchIdx);
                    getLoc(loc.begin(), vPriorbox[i].get(), COORDINATE_NUM * perPriorBoxNum[i], PriorBoxProperty_[i].width,stride,\
                            PriorBoxProperty_[i].height,boxInAnchor[i], vv_box,PriorBoxProperty_[i].variance);
                }
            }
            for (int i = 1; i < ClassNum; i++) {
                nms(vv_box[i], NmsThresh);
            }
            int box_num = 0;
            for (int i = 1; i < ClassNum; i++) {
                for (unsigned int j = 0; j < vv_box[i].size(); j++) {
                    if (vv_box[i][j].is_del) continue;
                    box_num++;
                }
            }

            output.resize(box_num*7);
            float *pdata=&(output[0]);
            for (int i = 1; i < ClassNum; i++) {
                for (unsigned int j = 0; j < vv_box[i].size(); j++) {
                    if (vv_box[i][j].is_del) continue;
                    *pdata = batchIdx;
                    pdata++;
                    *pdata = vv_box[i][j].type;
                    pdata++;
                    *pdata = vv_box[i][j].confidence;
                    pdata++;
                    *pdata  = (float) vv_box[i][j].left / OrigImgWidth;
                    pdata++;
                    *pdata  = (float) vv_box[i][j].top / OrigImgHeidth;
                    pdata++;
                    *pdata  = (float) vv_box[i][j].right / OrigImgWidth;
                    pdata++;
                    *pdata  = (float) vv_box[i][j].bottom / OrigImgHeidth;
                    pdata++;
                }
            }
        }

    private:
        void ssdCatOutput(VegaBaseInferOpCtx *ctx, unsigned int batchIdx,
                          std::vector<std::vector<SsdBox>> &vv_box, void *op) {
            DataBlob<T> loc_data = ctx->getBlob<T>(LocCatLayer, batchIdx);
            DataBlob<T> conf_data = ctx->getBlob<T>(ConfCatLayer, batchIdx);
            std::vector<int> loc_dim;
            (*getEnginOutputLayerDim)(LocCatLayer, loc_dim, op);
            std::vector<int> conf_dim;
            (*getEnginOutputLayerDim)(ConfCatLayer, conf_dim, op);
            //dim 0:c 1:h 2:w
            if ((conf_dim.size() < 3) || (loc_dim.size() < 3)) {
                LOG(ERROR) << "confidence's or locate's dimension erro ,conf dim size=" << conf_dim.size()
                           << " loc dim size=" << loc_dim.size();
                return;
            }
            //c =1 or w=1,else false
            if (!(conf_dim[0] == 1 || conf_dim[2] == 1)) {
                LOG(ERROR) << "confidence c=" << conf_dim[0] << " w=" << conf_dim[2];
                return;
            }
            if (!(loc_dim[0] == 1 || loc_dim[2] == 1)) {
                LOG(ERROR) << "location c=" << loc_dim[0] << " w=" << loc_dim[2];
                return;
            }
            int conf_num = toTalPriorBoxSize * ClassNum;
            std::vector<int> idx;
            int conf_stride = ClassNum;
            int conf_elem_size = 1;
            //dim 0:c 1:h 2:w
            if (conf_dim[2] == 1) {
                conf_stride = ClassNum * EngineAlign / sizeof(int);
                conf_num *= EngineAlign / sizeof(int);
                conf_elem_size = 1 * EngineAlign / sizeof(int);
            }
            for (int k = 1; k < ClassNum; k++) {
                T Thresh = ConfThresh[k];
                for (int i = k * conf_elem_size; i < conf_num; i += conf_stride) {
                    if (conf_data[i] > Thresh) {
                        idx.push_back((i - k * conf_elem_size) / conf_stride);
                    }
                }
                for (unsigned int i = 0; i < idx.size(); i++) {
                    unsigned int j = 0;
                    for (; j < prior_size_step.size(); j++) {
                        if (idx[i] < prior_size_step[j])    //caculate which prior_box the idx belong to ;
                        {
                            break;
                        }
                    }

                    int prior_ofset = 0;
                    if (j > 0) {
                        prior_ofset = idx[i] - prior_size_step[j - 1];
                    } else {
                        prior_ofset = idx[i];
                    }
                    SsdBox box;
                    box.is_del = false;
                    //dim 0:c 1:h 2:w
                    int stride = 1;
                    if (loc_dim[2] == 1) {
                        stride = EngineAlign / sizeof(int);
                    }
                    locToBox(loc_data.begin(), idx[i] * COORDINATE_NUM * stride, 0, stride,
                             vPriorbox[j].get() + prior_ofset * COORDINATE_NUM, box, PriorBoxProperty_[j].variance);
                    box.type = k;
                    box.confidence =  conf_data[idx[i] * conf_stride + k * conf_elem_size] ;
                    if(typeid(type_id)==typeid(int))
                    {
                        box.confidence/=4096.0f;
                    }
                    vv_box[k].push_back(box);
                }
            }
        }

        void getConfScore(T * data_in, int prior_num, int width, int stride, int height, std::vector<BoxInAnchor> &vboxInAnchor) {
            int in_offet = 0;
            int ch_size = stride * height;
            T pos_conf[ClassNum];
            int ch_offset;
            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pos = in_offet + j;
                    ch_offset = 0;
                    for (int n = 0; n < prior_num; n++) {
                        for (int k = 0; k < ClassNum; k++) {
                            pos_conf[k] = data_in[pos + ch_offset];
                            ch_offset += ch_size;
                        }
                        softMax(pos_conf, ClassNum, vboxInAnchor,j,i,n);
                    }
                }
                in_offet = in_offet + stride;
            }
        }
        void getConfScoreClassNum2(T *data_in, int prior_num, int width, int stride, int height,
                                   std::vector<BoxInAnchor> &vboxInAnchor ) {
            int in_offet = 0;
            int ch_size = stride * height;
            int ch_size_2 = ch_size * 2;
            T pos_conf[2];
            int ch_offset;
            T thres = LogConfThresh[1] ;
            float scoreThres=FConfThresh[1];

            for (int i = 0; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pos = in_offet + j;
                    ch_offset = 0;
                    for (int n = 0; n < prior_num; n++) {
                        if (data_in[pos + ch_offset + ch_size] - data_in[pos + ch_offset] > thres) {
                            pos_conf[0] = data_in[pos + ch_offset];
                            pos_conf[1] = data_in[pos + ch_offset + ch_size];
                            T max = pos_conf[0] > pos_conf[1] ? pos_conf[0] : pos_conf[1];
                            float v_ex[2];
                            if(typeid(type_id)==typeid(int)) {
                                v_ex[0] = exp((float) (pos_conf[0] - max) / 4096.0f);
                                v_ex[1] = exp((float) (pos_conf[1] - max) / 4096.0f);
                            } else{
                                v_ex[0] = exp(pos_conf[0] - max );
                                v_ex[1] = exp(pos_conf[1] - max );
                            }
                            float score= v_ex[1] / (v_ex[0] + v_ex[1]);

                            if(score>scoreThres)
                            {
                                BoxInAnchor boxInAnchor ;
                                boxInAnchor.x=j;
                                boxInAnchor.y=i;
                                boxInAnchor.prior_num=n;
                                boxInAnchor.conf=score;
                                boxInAnchor.type=1;
                                vboxInAnchor.push_back(boxInAnchor);
                            }
                        }
                        ch_offset += ch_size_2;
                    }
                }
                in_offet = in_offet + stride;
            }
        }



        void getLoc(T * data_loc, int *prior_box, int ch_n, int width, int stride, int height,std::vector<BoxInAnchor> vboxInAnchor,
                std::vector<std::vector<SsdBox>> &vv_box, std::vector<float> variance) {
            int ch_size = stride * height;
            for(unsigned int i=0;i<vboxInAnchor.size();i++)
            {
                SsdBox box;
                box.is_del = false;
                box.type =vboxInAnchor[i].type ;
                int pos = vboxInAnchor[i].y*stride + vboxInAnchor[i].x;
                int base_box_sn=(vboxInAnchor[i].y*width+vboxInAnchor[i].x)*ch_n+vboxInAnchor[i].prior_num * COORDINATE_NUM ;
                locToBox(data_loc, pos, vboxInAnchor[i].prior_num * COORDINATE_NUM, ch_size,
                         prior_box + base_box_sn , box, variance);
                box.confidence = vboxInAnchor[i].conf;
                vv_box[box.type].push_back(box);
            }

        }

        void locToBox(T * data_loc, int offset, int ch_star, int ch_size, int *prior_box, SsdBox &box,
                      std::vector<float> variance) {
            float prior_width, prior_height, prior_center_x, prior_center_y;
            prior_width = (float) (prior_box[2] - prior_box[0]);
            prior_height = (float) (prior_box[3] - prior_box[1]);
            prior_center_x = (prior_box[2] + prior_box[0]) * 0.5f;
            prior_center_y = (prior_box[3] + prior_box[1]) * 0.5f;
            float dec_box_center_x, dec_box_center_y, dec_box_center_h, dec_box_center_w;
            float loc0,loc1,loc2,loc3;
            if (typeid(type_id) == typeid(int)) {
                loc0 = (float) data_loc[offset + ch_star * ch_size] / 4096.0f;
                loc1 = (float) data_loc[offset + (ch_star + 1) * ch_size] / 4096.0f;
                loc2 = (float) data_loc[offset + (ch_star + 2) * ch_size] / 4096.0f;
                loc3 = (float) data_loc[offset + (ch_star + 3) * ch_size] / 4096.0f;
            } else{
                loc0 = data_loc[offset + ch_star * ch_size] ;
                loc1 = data_loc[offset + (ch_star + 1) * ch_size] ;
                loc2 = data_loc[offset + (ch_star + 2) * ch_size] ;
                loc3 = data_loc[offset + (ch_star + 3) * ch_size] ;
            }

            dec_box_center_x = variance[0] * loc0 * prior_width + prior_center_x;
            dec_box_center_y = variance[1] * loc1 * prior_height + prior_center_y;
            dec_box_center_w = exp(variance[2] * loc2) * prior_width;
            dec_box_center_h = exp(variance[3] * loc3) * prior_height;

            box.left = (int) (dec_box_center_x - dec_box_center_w * 0.5);
            box.top = (int) (dec_box_center_y - dec_box_center_h * 0.5);
            box.right = (int) (dec_box_center_x + dec_box_center_w * 0.5);
            box.bottom = (int) (dec_box_center_y + dec_box_center_h * 0.5);
        }

        inline void softMax(T *conf, int class_num, std::vector<BoxInAnchor> &vboxInAnchor, int x, int y, int prior_box_n) {

            T conf_sum = 0;
            T conf_avg ;
            for (int k = 0; k < ClassNum; k++) {
                conf_sum += conf[k];
            }
            conf_avg = conf_sum / ClassNum;
            int k=1;
            for (; k < ClassNum; k++) {
                if (conf[k] - conf_avg > LogConfThresh[k]) {
                    break;
                }
            }
            if (k==ClassNum) {
                return;
            }
            T max_conf = 0;
            for (int k = 0; k < ClassNum; k++) {
                if (conf[k] > max_conf) {
                    max_conf = conf[k];
                }
            }
            float e_sum = 0;
            std::vector<float> v_ex;
            v_ex.resize(ClassNum);
            for (int k = 0; k < ClassNum; k++) {
                if (typeid(type_id) == typeid(int))
                    v_ex[k] = exp((float) (conf[k] - max_conf) / 4096.0f);
                else
                    v_ex[k] = exp(conf[k] - max_conf);
                e_sum += v_ex[k];
            }
            for (int k = 1; k < ClassNum; k++) {
                float score = v_ex[k] / e_sum;
                if (score > FConfThresh[k]) {
                    BoxInAnchor boxInAnchor;
                    boxInAnchor.x = x;
                    boxInAnchor.y = y;
                    boxInAnchor.prior_num = prior_box_n;
                    boxInAnchor.conf = score;
                    boxInAnchor.type = k;
                    vboxInAnchor.push_back(boxInAnchor);
                }
            }
        }


        DgError getProperties(std::shared_ptr<ModelConfig> config_,std::string ssd_config_path) {
            SsdConfig ssdConfig;
            ssdConfig.load(config_->config_path_ + "/" + ssd_config_path, ClassNum, NmsThresh, PriorBoxProperty_);
            SSDSplitType = ssdConfig.getSplitType();
            ConfCatLayer = ssdConfig.getConfCatLayer();
            LocCatLayer = ssdConfig.getLocCatLayer();
            return DG_OK;
        }

        void priorBoxInit(int *out_put, SsdConfig::PriorBoxProperty &prior_box_para, int &num_prior) {
            for (unsigned int i = 0; i < prior_box_para.aspect_ratio.size(); i++) {
                if (prior_box_para.flip) {
                    prior_box_para.aspect_ratio[i] = 1.0f / prior_box_para.aspect_ratio[i];
                }
            }
            int index = 0;
            for (int h = 0; h < prior_box_para.height; h++) {
                for (int w = 0; w < prior_box_para.width; w++) {
                    float center_x = (w + prior_box_para.offset) * prior_box_para.step_w;
                    float center_y = (h + prior_box_para.offset) * prior_box_para.step_h;
                    for (unsigned int n = 0; n < prior_box_para.min_size.size(); n++) {
                        /*** first prior ***/
                        float box_height = prior_box_para.min_size[n];
                        float box_widht = prior_box_para.min_size[n];
                        out_put[index++] = (int) (center_x - box_widht * 0.5);
                        out_put[index++] = (int) (center_y - box_height * 0.5);
                        out_put[index++] = (int) (center_x + box_widht * 0.5);
                        out_put[index++] = (int) (center_y + box_height * 0.5);
                        /*** second prior ***/
                        if (prior_box_para.max_size.size() > n) {
                            float max_box_width = sqrt(prior_box_para.min_size[n] * prior_box_para.max_size[n]);
                            box_height = max_box_width;
                            box_widht = max_box_width;
                            out_put[index++] = (int) (center_x - box_widht * 0.5);
                            out_put[index++] = (int) (center_y - box_height * 0.5);
                            out_put[index++] = (int) (center_x + box_widht * 0.5);
                            out_put[index++] = (int) (center_y + box_height * 0.5);
                        }
                        /******aspect prior ***********/
                        for (unsigned int i = 0; i < prior_box_para.aspect_ratio.size(); i++) {
                            box_widht = (float) (prior_box_para.min_size[n] * sqrt(prior_box_para.aspect_ratio[i]));
                            box_height = (float) (prior_box_para.min_size[n] / sqrt(prior_box_para.aspect_ratio[i]));
                            out_put[index++] = (int) (center_x - box_widht * 0.5);
                            out_put[index++] = (int) (center_y - box_height * 0.5);
                            out_put[index++] = (int) (center_x + box_widht * 0.5);
                            out_put[index++] = (int) (center_y + box_height * 0.5);
                        }
                    }
                }
            }
            /***clip the out side of imge***********/
            if (prior_box_para.clip) {
                for (unsigned int i = 0; i <
                                         (unsigned int) (prior_box_para.width * prior_box_para.height * COORDINATE_NUM *
                                                         num_prior / 2); i++) {
                    out_put[2 * i] = MIN((unsigned int) MAX(out_put[2 * i], 0), OrigImgWidth);
                    out_put[2 * i + 1] = MIN((unsigned int) MAX(out_put[2 * i + 1], 0), OrigImgHeidth);
                }
            }
        }

        static inline bool mycmp(SsdBox b1, SsdBox b2) {
            return b1.confidence > b2.confidence;
        }

        void nms(std::vector<SsdBox> &vbox, float threshold) {
            sort(vbox.begin(), vbox.end(), mycmp);
            for (int i = 0; i < (int) vbox.size(); i++) {
                if (vbox[i].is_del)
                    continue;
                for (int j = i + 1; j < (int) vbox.size(); j++) {

                    if (!vbox[i].is_del) {
                        cv::Rect_<float> rect1(vbox[i].left, vbox[i].top, vbox[i].right - vbox[i].left + 1,
                                               vbox[i].bottom - vbox[i].top + 1);
                        cv::Rect_<float> rect2(vbox[j].left, vbox[j].top, vbox[j].right - vbox[j].left + 1,
                                               vbox[j].bottom - vbox[j].top + 1);
                        cv::Rect_<float> intersect = rect1 & rect2;
                        float verlap = intersect.area() * 1.0f
                                       / (rect1.area() + rect2.area() - intersect.area());
                        if (verlap > threshold) {
                            vbox[j].is_del = true;
                        }
                    }
                }
            }
        }

    };
}

#endif //VEGAHISI_VEGA_SSD_AFTER_INFER_H
