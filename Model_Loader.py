from absenteeism_module import *

model = absenteeism_model('model', 'scaler')

model.load_and_clean_data('Absenteeism_new_data.csv')

model.predicted_outputs()

model.predicted_outputs().to_csv('Absenteeism_predictions.csv', index=False)






