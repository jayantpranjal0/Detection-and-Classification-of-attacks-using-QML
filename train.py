from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 12345
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
from qiskit.algorithms.state_fidelities import ComputeUncompute
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit.utils import QuantumInstance
from qiskit import Aer, transpile
from qiskit import Aer, QuantumCircuit
# from qiskit_ibm_runtime import Sampler
from qiskit.primitives import Sampler
from qiskit_ibm_runtime import QiskitRuntimeService

# from qiskit import IBMQ
#IBMQ.delete_accounts()
# IBMQ.save_account('')
# IBMQ.load_account()
#quantum_instance = QuantumInstance(Aer.get_backend('aer_simulator_matrix_product_state'),
#  shots=200)

#service = QiskitRuntimeService(channel="ibm_cloud")
#backend = service.backend("ibmq_qasm_simulator")
# backend = Aer.get_backend('aer_simulator_matrix_product_state')

adhoc_dimension=4
adhoc_feature_map = ZZFeatureMap(feature_dimension=adhoc_dimension, reps=2, entanglement="linear")
sampler = Sampler()
fidelity = ComputeUncompute(sampler=sampler)
adhoc_kernel = FidelityQuantumKernel(fidelity=fidelity, feature_map=adhoc_feature_map)
import pandas as pd
train_features=pd.read_csv("dataset_check.csv")
# train_features=train_features.drop_duplicates()
train_features
train_features=train_features.drop(['Unnamed: 0'],axis=1)
train_labels=train_features["Class"]
train_features=train_features.drop(["Class"],axis=1)
from qiskit_machine_learning.algorithms import QSVC
qsvc = QSVC(quantum_kernel=adhoc_kernel)
print("Model Started Training")
qsvc.fit(train_features, train_labels)
print("Model Trained")
# test_labels=test_features["Class"]
# test_features=test_features.drop(["Class"],axis=1)
# qsvc_score=qsvc.score(test_features,test_labels)
# qsvc_score = qsvc.score(train_features, train_labels)
# print(f"QSVC classification test score: {qsvc_score}")
from joblib import dump, load
dump(qsvc, 'model.joblib')


print("Accuracy on training data = ",qsvc.score(train_features,train_labels))