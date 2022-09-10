import numpy as np
import sys
import json
from tflite_runtime.interpreter import Interpreter

x = sys.argv[1]
dict_file = json.loads(x)
data = dict_file["feeds"][0]
data["field5"] = data["field5"][:-4]
input_data = [float(data["field1"]), float(data["field2"]),
  float(data["field3"]), float(data["field4"]), float(data["field5"])]
input_array = np.array(input_data, dtype = np.float32)
input_array = np.resize(input_array, (1,5))
tflite_interpreter = Interpreter(model_path='converted_model.tflite')

input_details = tflite_interpreter.get_input_details()
output_details = tflite_interpreter.get_output_details()
tflite_interpreter.allocate_tensors()

tflite_interpreter.set_tensor(input_details[0]['index'], input_array)
tflite_interpreter.invoke()
output_array = tflite_interpreter.get_tensor(output_details[0]['index'])
result = np.squeeze(output_array)
print(int(result > 0.6))
