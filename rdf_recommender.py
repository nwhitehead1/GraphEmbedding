import sys
import bz2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from pykg2vec.utils.kgcontroller import KnowledgeGraph, UserDefinedDataset
from pykg2vec.config.config import KGEArgParser
from model_config import transe_config
from pykg2vec.core.TransE import TransE
from pykg2vec.utils.trainer import Trainer


class KnowledgeDataLoader:

    def __init__(self, data_dir, negative_sampling='uniform', test_size=0.2):
        self._data_dir = Path(data_dir)
        self._negative_sampling = negative_sampling
        self._triples: np.array = []
        self._test_size = test_size

    @property
    def data_dir(self):
        return self._data_dir

    @property
    def triples(self):
        return self._triples

    @property
    def negative_sampling(self):
        return self._negative_sampling

    @property
    def test_size(self):
        return self._test_size

    def get_knowledge_graph(self):
        dataset_name = self.data_dir.stem if '.' not in self.data_dir.stem else self.data_dir.stem.split('.')[0]
        custom_dataset_path = self.data_dir.parent / dataset_name
        if not custom_dataset_path.exists():
            Path(custom_dataset_path).mkdir(parents=True, exist_ok=True)

        # Means directory is created, but split files don't exist
        if not custom_dataset_path.is_file():
            self.__load()
            self.__split(write_path=custom_dataset_path, name=dataset_name)
        return KnowledgeGraph(dataset=dataset_name, negative_sample=self.negative_sampling, custom_dataset_path=custom_dataset_path)

    # TODO: Extend this functionality for parsing multiple filetypes + streaming for large files
    # For loading large bz2 zipped n-triple files. Not extended for other files yet
    # Can be extended for different file types - need smarter parser for very large files
    def __load(self):
        # Temporary catchall for bad files
        acceptable_formats = ['.ttl', '.nt']
        if self.data_dir.suffixes[0] in acceptable_formats:
            triples = []
            with bz2.BZ2File(self.data_dir, 'rb') as file:
                for line in file:
                    decoded_line = line.decode('utf-8').split()
                    if decoded_line[0] == '#':
                        continue
                    triples.append((decoded_line[0], decoded_line[1], decoded_line[2]))
            self.triples.extend(triples)
        else:
            raise Exception('File extension is not supported.')

    def __split(self, write_path, name):
        x_train, x_test = train_test_split(self.triples, test_size=self.test_size)
        x_test, x_valid = train_test_split(x_test, test_size=0.5)
        self.__write(self, triples=x_test, write_path=write_path / Path(name+'-test.txt'))
        self.__write(self, triples=x_train, write_path=write_path / Path(name+'-train.txt'))
        self.__write(self, triples=x_valid, write_path=write_path / Path(name+'-valid.txt'))

    @staticmethod
    def __write(self, triples: np.array, write_path: Path):
        with open(write_path, 'w', encoding='utf-8') as f:
            for triple in triples:
                try:
                    f.write(str(triple[0]) + '\t' + str(triple[1]) + '\t' + str(triple[2]) + '\n')
                except UnicodeEncodeError:
                    continue


def main():

    args = KGEArgParser().get_args(sys.argv[1:])

    if Path(args.dataset_path).exists():
        kdl = KnowledgeDataLoader(data_dir=args.dataset_path, negative_sampling=args.sampling)
        kg = kdl.get_knowledge_graph()
        print('Successfully loaded {} triples from {}.'.format(len(kdl.triples), kdl.data_dir))
    else:
        print('Unable to find dataset from path:', args.dataset_path)
        print('Default loading Freebase15k dataset with default hyperparameters...')
        kg = KnowledgeGraph()

    kg.prepare_data()
    kg.dump()

    # TODO: Not sure why new dataset isn't cached on subsequent hits...
    args.dataset_path = './data/' + kg.dataset_name
    args.dataset_name = kg.dataset_name

    # Add new model configurations to run.
    models = [TransE(transe_config(args=args))]

    for model in models:
        print('---- Training Model: {} ----'.format(model.model_name))
        trainer = Trainer(model=model, debug=args.debug)
        trainer.build_model()
        trainer.train_model()
        tf.reset_default_graph()


# Custom Dataset with no tuning    :  python rdf_recommender.py -dsp "./data/topical_concepts_en.ttl.bz2" [-plote True]
# Freebase15k with Bayesian tuning :  python rdf_recommender.py -ghp True [-plote True]
if __name__ == "__main__":
    main()

