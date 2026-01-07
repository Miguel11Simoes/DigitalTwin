#ifndef TVMGEN_DEFAULT_H_
#define TVMGEN_DEFAULT_H_
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*!
 * \brief Input tensor spec size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_SPEC_SIZE 65536
/*!
 * \brief Input tensor aux size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_AUX_SIZE 408
/*!
 * \brief Output tensor output0 size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_OUTPUT0_SIZE 64
/*!
 * \brief Output tensor output3 size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_OUTPUT3_SIZE 4
/*!
 * \brief Output tensor output1 size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_OUTPUT1_SIZE 16
/*!
 * \brief Output tensor output2 size (in bytes) for TVM module "default" 
 */
#define TVMGEN_DEFAULT_OUTPUT2_SIZE 4
/*!
 * \brief Input tensor pointers for TVM module "default" 
 */
struct tvmgen_default_inputs {
  void* aux;
  void* spec;
};

/*!
 * \brief Output tensor pointers for TVM module "default" 
 */
struct tvmgen_default_outputs {
  void* output0;
  void* output1;
  void* output2;
  void* output3;
};

/*!
 * \brief entrypoint function for TVM module "default"
 * \param inputs Input tensors for the module 
 * \param outputs Output tensors for the module 
 */
int32_t tvmgen_default_run(
  struct tvmgen_default_inputs* inputs,
  struct tvmgen_default_outputs* outputs
);
/*!
 * \brief Workspace size for TVM module "default" 
 */
#define TVMGEN_DEFAULT_WORKSPACE_SIZE 4784640

#ifdef __cplusplus
}
#endif

#endif // TVMGEN_DEFAULT_H_
