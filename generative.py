import pickle
from deepproblog.model import Model
from deepproblog.logger import VerboseLogger

model = Model.from_file('generative.pl', logger=VerboseLogger(100))


model.fit()

with open('model.dpl', 'wb') as f:
    pickle.dump(model, f)

# with open('mnist.pl') as f:
#     queries = list(model.load_queries(f.read()))

# query = queries[0]
# engine = ExactEngine(model)
# ac = engine.query(query)
# result = ac.evaluate(model, substitution=query.substitution)
# print(result)
