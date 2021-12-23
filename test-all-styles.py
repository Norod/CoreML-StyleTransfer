import coremltools
from PIL import Image
import glob
import os

inputImage = Image.open('input.jpg')
output_folder = './results'
if os.path.exists(output_folder) == False:
    os.mkdir(output_folder)

models_folder= "./models"
model_paths = glob.glob(os.path.join(models_folder, "*.mlmodel"))
for model_path in model_paths:
    modelname_w_ext = os.path.basename(model_path)
    model_name, model_extension = os.path.splitext(modelname_w_ext)
    print("Loading \"" + model_name + "\"")
    model =  coremltools.models.MLModel(model_path)
    resized_image = inputImage.resize((512, 512))
    pred = model.predict({'image': resized_image})
    result = pred['stylizedImage']
    result = result.resize((inputImage.width, inputImage.height)).convert('RGB')
    result_file = model_name + '_stylized.jpg'
    result_path = os.path.join(output_folder, result_file)
    result.save(result_path)
    print("Saved result to " + result_path)