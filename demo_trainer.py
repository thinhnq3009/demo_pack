from schema.reader.json_reader import JsonReader
from tensor.trainer import ModelTrainer

trainer = ModelTrainer(name_model="gpt-1")

reader = JsonReader(path="template_data.json")

trainer.read_dateset(reader)
trainer.init_model()
trainer.compile_model(path="models")

