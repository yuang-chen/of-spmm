#ifndef ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_H_
#define ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_H_

#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/graph/op_graph.h"

namespace oneflow {

void AutoGrad(const OpGraph& op_graph, JobConf1* job_conf,
              HashMap<LogicalBlobId, LogicalBlobId>* lbi2diff_lbi);
void AddTotalLossInstanceNumOpConf(const OpGraph& op_graph, JobConf1* job_conf,
                                   LogicalBlobId* total_loss_instance_num_lbi);

void GenerateBackwardOpConfIf(const Operator&, std::vector<OperatorConf>*,
                              const std::function<LogicalBlobId*(const std::string&)>&);

class GenerateBackwardOpConfWrapperStruct final {
 public:
  using NaiveFunc = std::function<void(const Operator&, std::vector<OperatorConf>*,
                                       const std::function<LogicalBlobId*(const std::string&)>&)>;
  using Func = std::function<void(const Operator&, std::vector<OperatorConf>*,
                                  const std::function<LogicalBlobId*(const std::string&)>&,
                                  const std::function<const BlobDesc&(const std::string&)>&)>;
  GenerateBackwardOpConfWrapperStruct(const NaiveFunc& f)
      : naive_func_(std::make_unique<NaiveFunc>(f)) {}
  GenerateBackwardOpConfWrapperStruct(const Func& f) : func_(std::make_unique<Func>(f)) {}
  void Call(const Operator&, std::vector<OperatorConf>*,
            const std::function<LogicalBlobId*(const std::string&)>&,
            const std::function<const BlobDesc&(const std::string&)>&) const;

 private:
  const std::unique_ptr<const NaiveFunc> naive_func_;
  const std::unique_ptr<const Func> func_;
};

#define REGISTER_OP_GRAD(op_type_case, gen_grad_func)                       \
  REGISTER_CLASS_CREATOR(op_type_case, GenerateBackwardOpConfWrapperStruct, \
                         ([] { return new GenerateBackwardOpConfWrapperStruct(gen_grad_func); }))

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTOGRAD_AUTOGRAD_H_
