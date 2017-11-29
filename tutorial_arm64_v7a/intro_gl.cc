#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>

// #include "caffe2/mobile/contrib/opengl/android/AndroidGLContext.h"
// #include "caffe2/mobile/contrib/opengl/android/arm_neon_support.h"
// #include "caffe2/mobile/contrib/opengl/android/gl3stub.h"
// #include "caffe2/mobile/contrib/opengl/core/GL.h"
#include "caffe2/mobile/contrib/opengl/core/GLPredictor.h"

CAFFE2_DEFINE_string(init_net, "res/squeezenet_init_net.pb", "The give path to the init protobuffer");
CAFFE2_DEFINE_string(predict_net, "res/squeezenet_predict_net.pb", "The give path to the predict protobuffer");
CAFFE2_DEFINE_string(image_file, "res/image.jpg", "The image file");
CAFFE2_DEFINE_int(size, 227, "The image size");

namespace caffe2 {

void print(const Blob* blob, const std::string& name) {
  auto tensor = blob->Get<TensorCPU>();
  const auto& data = tensor.data<float>();
  std::cout << name << "(" << tensor.dims()
            << "): " << std::vector<float>(data, data + tensor.size())
            << std::endl;
}

void run() {
  const bool &test = false;
  std::cout << std::endl;
  std::cout << "## Caffe2 Intro Tutorial ##" << std::endl;
  std::cout << "https://caffe2.ai/docs/intro-tutorial.html" << std::endl;
  std::cout << std::endl;

  // >>> from caffe2.python import workspace, model_helper
  // >>> import numpy as np
  Workspace workspace;

  // >>> x = np.random.rand(4, 3, 2)
  std::vector<float> x(4 * 3 * 2);
  for (auto& v : x) {
    v = (float)rand() / RAND_MAX;
  }

  // >>> print(x)
  std::cout << x << std::endl;

  // >>> workspace.FeedBlob("my_x", x)
  {
    auto tensor = workspace.CreateBlob("my_x")->GetMutable<TensorCPU>();
    auto value = TensorCPU({4, 3, 2}, x, NULL);
    tensor->ResizeLike(value);
    tensor->ShareData(value);
  }

  // >>> x2 = workspace.FetchBlob("my_x")
  // >>> print(x2)
  {
    const auto blob = workspace.GetBlob("my_x");
    print(blob, "my_x");
  }
  
}

void copy_to_opengl(NetDef &netdef, 
                    const std::string &data_cpu, 
                    const std::string &data_opengl)
{
  auto op = netdef.add_op();
  op->set_type("CopyToOpenGL");
  op->add_input(data_cpu);
  op->add_output(data_opengl);
}

void copy_from_opengl(NetDef &netdef, 
                      const std::string &data_cpu, 
                      const std::string &data_opengl)
{
  auto op = netdef.add_op();
  op->set_type("CopyFromOpenGL");
  op->add_input(data_opengl);
  op->add_output(data_cpu);
}

void fill_data(Workspace &workspace, 
               const std::string &blob_name,
               const int &K = 0, 
               const int &C = 0,
               const int &H = 0, 
               const int &W = 0)
{
  auto tensor = workspace.CreateBlob(blob_name)->GetMutable<TensorCPU>();
  if(C != 0)
    tensor->Resize(K, C, H, W);
  else
    tensor->Resize(K);
  float *data = tensor->mutable_data<float>();
  for(int i = 0; i < tensor->size(); ++i)
    data[i] = (float)rand() / RAND_MAX;
}

void add_conv(Workspace &workspace,
              NetDef &netdef,
              const std::string &inputs, 
              const std::string &outputs,
              const int &K, 
              const int &C,
              const int &H, 
              const int &W,
              const int &kernel = 3, 
              const int &stride = 2,
              const int &pad = 0,
              const std::string &order = "NCHW")
{
  fprintf(stderr, "add convolution layer\n");
  {
    fill_data(workspace, inputs + "_conv_weights", K, C, H, W);
    fill_data(workspace, inputs + "conv_bias", K);
  }
  {
    auto op = netdef.add_op();
    op->set_type("OpenGLConv");
    op->add_input(inputs);
    op->add_input(inputs + "_conv_weights");
    op->add_input(inputs + "conv_bias");
    
    {
      auto arg = op->add_arg();
      arg->set_name("order");
      arg->set_s(order);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("kernel");
      arg->set_i(kernel);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("pad");
      arg->set_i(pad);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("stride");
      arg->set_i(stride);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("is_last");
      arg->set_i(1);
    }
    
    op->add_output(outputs);
  }
}

void add_maxpool(Workspace &workspace,
                 NetDef &netdef,
                 const std::string &inputs,
                 const std::string &outputs,
                 const int &kernel = 2,
                 const int &stride = 2,
                 const int &pad = 0,
                 const std::string &order = "NCHW")
{
  fprintf(stderr, "add maxpool layer\n");
  {
    auto op = netdef.add_op();
    op->set_type("OpenGLMaxPool");
    op->add_input(inputs);
    
    {
      auto arg = op->add_arg();
      arg->set_name("order");
      arg->set_s(order);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("kernel");
      arg->set_i(kernel);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("pad");
      arg->set_i(pad);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("stride");
      arg->set_i(stride);
    }
    
    {
      auto arg = op->add_arg();
      arg->set_name("is_last");
      arg->set_i(1);
    }
    
    op->add_output(outputs);
  }
}

void laMiaRun()
{
  std::cerr << std::endl;
  std::cerr << "## Caffe2 example to run with OpenGL ##" << std::endl;
  std::cerr << std::endl;
  
  const bool use_opengl = false;
  
  fprintf(stderr, "construct network\n");
  NetDef netdef;
  Workspace workspace;
  
  fprintf(stderr, "create inputs tensor\n");
  {
    fill_data(workspace, "inputs_cpu", 1, 3, 29, 29);
    // convert to OpenGL from CPU
    copy_to_opengl(netdef, "inputs_cpu", "inputs");
  }
  
  // add Conv and Maxpool
  {
    add_conv(workspace, netdef, "inputs", "conv1", 3, 3, 3, 3);
    add_maxpool(workspace, netdef, "conv1", "pool1");
    add_conv(workspace, netdef, "pool1", "conv2", 1, 3, 3, 3);
    add_conv(workspace, netdef, "conv2", "conv3", 1, 1, 3, 3);
    add_conv(workspace, netdef, "conv3", "conv4", 1, 1, 1, 1, 1, 1);
    add_conv(workspace, netdef, "conv4", "conv5", 1, 1, 1, 1, 1, 1);
  } 
  
  // add Softmax
  {
    auto op = netdef.add_op();
    op->set_type("OpenGLSoftmax");
    op->add_input("conv5");
    op->add_output("outputs");
  }
  
  // fetch ftom OpenGL to CPU
  copy_from_opengl(netdef, "outputs_cpu", "outputs");

do{  
  fprintf(stderr, "run network\n");
  double start_time = clock();
  workspace.RunNetOnce(netdef);
  double stop_time = clock();
  
  fprintf(stderr, "fetch outputs\n");
  const auto blob = workspace.GetBlob("outputs_cpu");
  print(blob, "outputs");
  
  fprintf(stderr, "time cost: %f seconds\n", (stop_time - start_time) / CLOCKS_PER_SEC);
}while(true);  
}

/*
void laMiaRun(const std::string &model_file)
{
  std::cerr << std::endl;
  std::cerr << "## Caffe2 example to run an existing model with OpenGL ##" << std::endl;
  std::cerr << std::endl;
  
  NetDef init_net, predict_net;
  
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_init_net, &init_net));
  CAFFE_ENFORCE(ReadProtoFromFile(FLAGS_predict_net, &predict_net));
  
  GLPredictor p(init_net, predict_net);
  
  std::vector<float> data(1 * 3 * 227 * 227);
  for(auto &v : data)
    v = (float)rand() / RAND_MAX;
  // TensorCPU input({1, 3, 227, 227}, data, NULL);
  // GLPredictor::TensorVector inputVec({&input}), outputVec;
  
  GLImage<float> single_input;
  GLImageVector<float> input;
  // input.push_back(&single_input);
  std::vector<GLImageVector<float>*> inputVec, outputVec;
do{
  double start_time = clock();
  p.run<float>(inputVec, &outputVec);
  double stop_time = clock();
  auto &output = *(outputVec[0]);

  // sort top results
  const auto &probs = output.data<float>();
  std::vector<std::pair<int, int>> pairs;
  for (auto i = 0; i < output.size(); i++) {
    if (probs[i] > 0.01) {
      pairs.push_back(std::make_pair(probs[i] * 100, i));
    }
  }

  std::sort(pairs.begin(), pairs.end());

  std::cout << std::endl;

  // show results
  std::cout << "output: " << std::endl;
  for (auto pair : pairs) {
    std::cout << "  " << pair.first << "% '" << pair.second
              << "' (" << pair.second << ")" << std::endl;
  }
  fprintf(stderr, "time cost: %f seconds\n", (stop_time - start_time) / CLOCKS_PER_SEC);
}while(true);
}
*/

}  // namespace caffe2

int main(int argc, char** argv) {
  std::cout << "Hey laMia!~\n";
  if(caffe2::GlobalInit(&argc, &argv))
    std::cout << "launch caffe2::GlobalInit(&argc, &argv) done\n";
  else
    std::cout << "launch caffe2::GlobalInit(&argc, &argv), failed\n";
  caffe2::laMiaRun();
  std::cout << "has cuda runtime: " << caffe2::HasCudaRuntime() << "\n";
  std::cout << "launch caffe2::run() done\n";
  google::protobuf::ShutdownProtobufLibrary();
  std::cout << "google::protobuf::ShutdownProtobufLibrary()\n";
  return 0;
}
