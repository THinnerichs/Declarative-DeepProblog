import pickle

from problog.logic import Term, Var, Constant
from torchvision.utils import save_image

from deepproblog.engines import ExactEngine
from deepproblog.query import Query
from deepproblog.model import Model
from deepproblog.logger import VerboseLogger
from MNIST.data import MNIST
from sklearn.metrics import accuracy_score
train = False

if train:
    model = Model.from_file('generative.pl', logger=VerboseLogger(1000))
    model.fit(batch_size=4, stop_condition=1)
    with open('model.dpl', 'wb') as f:
        pickle.dump(model, f)
else:

    with open('model.dpl', 'rb') as f:
        model = pickle.load(f)

mnist_test = MNIST('mnist_test')

engine = ExactEngine(model)

predictions = model.predict(mnist_test)
gt = list(mnist_test.get_label_indicators())
accuracy = accuracy_score(gt, predictions)
print('Accuracy', accuracy)

# query = Query(Term('digit', Term('tensor', Term('mnist_test', Constant(0))), Var('X')))
# ac = engine.query(query)
# results = ac.evaluate(model)
# for key in results:
#     print(key, ':{:.1e}'.format(float(results[key])))
# print('-'*20)
# total_probability = sum(float(results[key]) for key in results)
# for key in results:
#     print(key, ':{:.1e}'.format(float(results[key]) / total_probability))
#
#
# query = Query(Term('digit', Var('I'), Var('D')))
#
# ac = engine.query(query)
# results = ac.evaluate(model)
# for key in results:
#     tensor, label = key.args
#     print(key, ':', float(results[key]))
#     tensor = model.get_tensor(tensor).detach()
#     save_image(tensor, '{}_{}.png'.format(train, key), value_range=(-1.0, 1.0))
