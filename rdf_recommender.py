import sys
import bz2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.config import KGEArgParser
from model_config import transe_config, ntn_config
from pykg2vec.core.TransE import TransE
from pykg2vec.core.NTN import NTN
from pykg2vec.utils.trainer import Trainer


class KnowledgeDataLoader:

    def __init__(self, data_dir, test_size=0.2):
        self._data_dir = Path(data_dir).resolve()
        self._triples: np.array = []
        # this will change later
        self._name = None if self._data_dir is None else self._data_dir.name.rsplit('/', 1)[-1].rsplit('.', 2)[0]
        self._test_size = test_size
        self.load()
        self.split()

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def triples(self):
        return self._triples

    @property
    def name(self):
        return self._name

    @staticmethod
    def write(self, triples: np.array, write_path: Path):
        if not write_path.parent.exists():
            Path(write_path.parent).mkdir(parents=True, exist_ok=True)
        with open(write_path, 'w', encoding='utf-8') as f:
            for triple in triples:
                try:
                    f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
                except UnicodeEncodeError:
                    continue

    # For loading large n-triple files. Not extended for other files yet
    # Can be extended for different file types - need smarter parser for very large files
    def load(self):
        if self._data_dir.suffixes[0] == '.ttl':
            triples = []
            with bz2.BZ2File(self._data_dir, 'rb') as file:
                for line in file:
                    decoded_line = line.decode('utf-8').split()
                    if decoded_line[0] == '#':
                        continue
                    triples.append((decoded_line[0], decoded_line[1], decoded_line[2]))
            self._triples.extend(triples)
        else:
            raise Exception('File extension is not supported.')

    def split(self):
        x_train, x_test = train_test_split(self._triples, test_size=self._test_size)
        x_test, x_valid = train_test_split(x_test, test_size=0.5)
        self.write(self, triples=x_test, write_path=Path('./data/'+self._name+'/'+self._name+'-test.txt'))
        self.write(self, triples=x_train, write_path=Path('./data/'+self._name+'/'+self._name+'-train.txt'))
        self.write(self, triples=x_valid, write_path=Path('./data/'+self._name+'/'+self._name+'-valid.txt'))


def main():

    args = KGEArgParser().get_args(sys.argv[1:])

    if args.dataset_path is None:
        kg = KnowledgeGraph()
        kg.prepare_data()
        kg.dump()
    else:
        kgl = KnowledgeDataLoader(data_dir=args.dataset_path)
        print('Successfully loaded {} triples from {}.'.format(len(kgl.triples), kgl.data_dir))

        args.dataset_path = './data/' + kgl.name
        args.dataset_name = kgl.name

        # Define knowledge graph
        kg = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling,
                            custom_dataset_path=args.dataset_path)
        kg.prepare_data()
        kg.dump()

    models = [TransE(transe_config(args=args)), NTN(ntn_config(args=args))]

    for model in models:
        print('---- Training Model: {} ----'.format(model.model_name))
        trainer = Trainer(model=model, debug=args.debug)
        trainer.build_model()
        trainer.train_model()
        tf.reset_default_graph()


# Custom Dataset with no tuning   :     python rdf_recommender.py -dsp "./data/topical_concepts_en.ttl.bz2" -plote True
# Freebase15k with Bayesian tuning:     python rdf_recommender.py -ghp True
if __name__ == "__main__":
    main()

