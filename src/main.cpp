
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>

void decode(std::string, std::vector<float>&, int, int);
void get_image(std::string, int, int, float*);
void save_predict(std::string, int*, int, int);
void inference(std::string, std::string, std::string, std::string);

static int orgH, orgW;
int main(int argc, const char** argv)
{
    if(argc < 3)
    {
        fprintf(stderr, "Usage: ./segment infer/test/print\n");
        return -1;
    }

    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
        args.emplace_back(argv[i]);

    if (args[0] == "infer")
    {
        fprintf(stdout, "Usage: ./segment infer <path of xml> <path of img_in> <path of img_out>");
        const std::string xmlpth{args[1]};
        const std::string device = "CPU";
        std::string impth{args[2]};
        std::string savepth{args[3]};
        inference(xmlpth, device, impth, savepth);
    }    
    return 0;
}

void inference(std::string xml_path, std::string device, std::string img_in, std::string img_out) 
{
    // model setup
    std::cout << "load network: " << xml_path << std::endl;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(xml_path);

    // Configure input
    InferenceEngine::InputsDataMap inputs = model.getInputsInfo();
    InferenceEngine::InputInfo::Ptr input_info = inputs.begin()->second;
    std::string input_name = inputs.begin()->first;

    input_info->setPrecision(InferenceEngine::Precision::FP32);
    input_info->setLayout(InferenceEngine::Layout::NCHW);
    input_info->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);

    // Configure output
    if (model.getOutputsInfo().empty()) {
        std::cerr << "Network outputs info is empty" << std::endl;
    }
    InferenceEngine::DataPtr output_info = model.getOutputsInfo().begin()->second;
    output_info->setPrecision(InferenceEngine::Precision::I64);
    std::string output_name = model.getOutputsInfo().begin()->first;

    // Load model to the device
    InferenceEngine::ExecutableNetwork network = ie.LoadNetwork(model, device);

    // Create an infer request
    InferenceEngine::InferRequest infer_request = network.CreateInferRequest();

    // set input data    
    std::cout << "set input data from: " << img_in << std::endl;
    auto insize = input_info->getTensorDesc().getDims();
    int in_height = insize[2], in_width = insize[3];
    
    std::vector<float> seq;
    decode(img_in, seq, in_height, in_width);
    InferenceEngine::TensorDesc tDesc(InferenceEngine::Precision::FP32,
                                        {1, 3, insize[2], insize[3]},
                                        InferenceEngine::Layout::NCHW);
    InferenceEngine::Blob::Ptr inBlob = InferenceEngine::make_shared_blob<float>(tDesc, seq.data());
    infer_request.SetBlob(input_name, inBlob);

    // Do inference
    std::cout << "do inference " << std::endl;
    infer_request.Infer();


    // fetch output data
    std::cout << "save result to: " << img_out << std::endl;
    
    InferenceEngine::SizeVector outsize = output_info->getTensorDesc().getDims();
    if (outsize.empty() || !outsize[0])
    {
        throw std::runtime_error("Invalid output dimensions");
    }  

    InferenceEngine::Blob::Ptr outblob = infer_request.GetBlob(output_name);
    InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outblob);
    if (!moutput)
    {
        throw std::runtime_error("Output blob should be inherited from MemoryBlob");
    }

    auto moutputHolder = moutput->rmap();
    //using myBlobType = InferenceEngine::PrecisionTrait<InferenceEngine::Precision::U8>::value_type;
    int* p_outp = moutputHolder.as<int*>();
    int out_height = outsize[2], out_width = outsize[3];

    save_predict(img_out, p_outp, out_height, out_width);
}

void decode(std::string impth, std::vector<float>& sequence, int iH, int iW)
{
    const std::size_t framePixNum = 3 * iH* iW;
    const std::size_t frameByte = framePixNum * sizeof(float);
    sequence.resize(framePixNum);
    memset((void*)sequence.data(), 0, frameByte);
    
    cv::Mat im = cv::imread(impth);
    if (im.empty()) 
    {
        std::cerr << "cv::imread failed: " << impth << std::endl;
        std::abort();
    }

    //resize
    cv::Mat scaled;
    orgH = im.rows, orgW = im.cols;
    if ((orgH != iH) || orgW != iW) 
    {
        fprintf(stdout, "resize original size %dx%d to %dx%d according to model requirement.\n",orgH, orgW, iH, iW);      
        cv::resize(im, scaled, cv::Size(iW, iH), 0, 0, cv::INTER_LINEAR);
    }

    // convert type
    cv::Mat scaled_f;
    scaled.convertTo(scaled_f, CV_32FC3, 1 / 255.0);

    // split to BGR
    std::vector<cv::Mat> bgrVec;
    cv::split(scaled_f, bgrVec);
    cv::Mat &b = bgrVec[0], &g = bgrVec[1], &r = bgrVec[2];

    // normalize
    float mean[3] = {0.5081f, 0.4480f, 0.4340f};
    float std[3] = {0.2822f, 0.2757f, 0.2734f};
    r = r - mean[0];
    g = g - mean[1];
    b = b - mean[2];
    r = r / std[0];
    g = g / std[1];
    b = b / std[2];

    // copy
    if (r.isContinuous() && g.isContinuous() && b.isContinuous())
    {
        float* pDst = sequence.data();

        // note: must in R->G->B order!!!
        const std::size_t chanOffset = framePixNum / 3;
        memcpy((void*)(pDst + 0 * chanOffset), r.data, chanOffset * sizeof(float));
        memcpy((void*)(pDst + 1 * chanOffset), g.data, chanOffset * sizeof(float));
        memcpy((void*)(pDst + 2 * chanOffset), b.data, chanOffset * sizeof(float));
    }
    else
    {
        std::cerr << "Image not continous, implement this!" << std::endl;
    }
}

std::unordered_map<uint8_t, cv::Vec3b> gPalette = { //in bgr order
    { 0, { 255, 255, 255 } },
    { 1, { 28, 28, 183 } },
    { 2, { 188, 204, 255 } },
    { 3, { 1, 1, 1 } },
    { 4, { 140, 20, 74 } },
    { 5, { 150, 150, 150 } },
    { 6, { 161, 71, 13 } },
    { 7, { 243, 150, 33 } },
    { 8, { 100, 96, 0 } },
    { 9, { 23, 127, 245 } },
    { 10, { 45, 192, 251 } },
    { 11, { 59, 235, 255 } },
    //{ 12, { 196, 249, 196 } },
    { 12, { 0, 81, 230 } }
};

void save_predict(std::string savename, 
                  int* data, 
                  int oH,
                  int oW) 
{
    cv::Mat pred(cv::Size(oW, oH), CV_8UC3);
    for(int i = 0; i < oH; i++)
    {
        const int* head = data + i * oW;
        for(int j = 0; j < oW; j++)
        {
            uint8_t key = head[j];
            //assert(gPalette.find(key) != gPalette.end());
            if(gPalette.find(key) == gPalette.end())
            {
                fprintf(stdout, "change key from %d to 0\n", key);
                key = 0;
            }
            cv::Vec3b& p = pred.at<cv::Vec3b>(i,j);
            p = gPalette[key];
        }
    }

    cv::Mat pred_ori_size;
    if((orgH != oH) || (orgW != oW))
    {
        fprintf(stdout, "resize output size: %dx%d, to origin size: %dx%d\n", oH, oW, orgH, orgW);
        cv::resize(pred, pred_ori_size, cv::Size(orgW, orgH), 0, 0, cv::INTER_NEAREST);
    }
    cv::imwrite(savename, pred_ori_size);
}