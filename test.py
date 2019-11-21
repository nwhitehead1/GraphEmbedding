import sys
import bz2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from pykg2vec.utils.kgcontroller import KnowledgeGraph
from pykg2vec.config.config import Importer, KGEArgParser
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
            print('Successfully loaded {} triples from {}'.format(len(triples), self._name))
            self._triples.extend(triples)
        else:
            raise Exception('File extension is not supported.')

    def split(self):
        x_train, x_test = train_test_split(self._triples, test_size=self._test_size)
        x_test, x_valid = train_test_split(x_test, test_size=0.5)
        self.write(self, triples=x_test, write_path=Path('./'+self._name+'/'+self._name+'-test.txt'))
        self.write(self, triples=x_train, write_path=Path('./'+self._name+'/'+self._name+'-train.txt'))
        self.write(self, triples=x_valid, write_path=Path('./'+self._name+'/'+self._name+'-valid.txt'))


def main():

    args = KGEArgParser().get_args(sys.argv[1:])

    # Getting triples in the appropriate format for pykg2vec
    if not Path(args.dataset_path).exists():
        KnowledgeDataLoader(data_dir='./topical_concepts_en.ttl.bz2')

    kg = KnowledgeGraph(dataset=args.dataset_name, negative_sample=args.sampling, custom_dataset_path=args.dataset_path)
    kg.prepare_data()
    kg.dump()

    # Configuration
    config_def, model_def = Importer().import_model_config(args.model_name.lower())
    config = config_def(args=args)
    model = model_def(config)

    # Create, Compile and Train the model. While training, several evaluation will be performed.
    trainer = Trainer(model=model, debug=args.debug)
    trainer.build_model()
    trainer.train_model()


# python test.py -mn TransE -ds topical_concepts_en -dsp "./topical_concepts_en"
if __name__ == "__main__":
    main()
