import os

from .general_utils import get_logger
from .data_utils import get_trimmed_wordvec_vectors, load_vocab, \
        get_processing_word


class Config():
    def __init__(self, parser, load=True):
        """Initialize hyperparameters and load vocabs

        Args:
            load_embeddings: (bool) if True, load embeddings into
                np array, else None

        """
        ## parse args
        self.parser = parser
        # training parameters
        parser.add_argument('--nepochs', default='40', type=int,
                    help='number of epochs')
        parser.add_argument('--dropout', default='0.5', type=float,
                    help='number of epochs')
        parser.add_argument('--batch_size', default='40', type=int,
                    help='batch size')
        parser.add_argument('--lr', default='0.001', type=float,
                    help='learning rate')
        parser.add_argument('--lr_method', default='adam', type=str,
                    help='optimization method')
        parser.add_argument('--lr_decay', default='0.99', type=float,
                    help='learning rate decay rate')
        parser.add_argument('--clip', default='10', type=float,
                    help='gradient clipping')
        parser.add_argument('--nepoch_no_imprv', default='3', type=int,
                    help='number of epoch patience')
        parser.add_argument('--l2_reg_lambda', default='0.0001', type=float,
                    help='l2 regularization coefficient')

        # data and results paths
        parser.add_argument('--dir_output', default='test', type=str,
                    help='directory for output')
        parser.add_argument('--data_root', default='/data/medg/misc/jindi/nlp/PICO', type=str,
                    help='directory for output')
        parser.add_argument('--filename_wordvec_trimmed', default='data/word2vec_pubmed.trimmed.txt', 
                    type=str, help='directory for trimmed word embeddings file')
        parser.add_argument('--filename_wordvec', default='/data/medg/misc/jindi/nlp/embeddings/word2vec/wikipedia-pubmed-and-PMC-w2v.txt', 
                    type=str, help='directory for original word embeddings file')

        # model hyperparameters
        parser.add_argument('--hidden_size_lstm_sentence', default='150', type=int,
                    help='hidden size of sentence level lstm')
        parser.add_argument('--attention_size', default='300', type=int,
                    help='attention vector size')

        # misc
        parser.add_argument('--restore', action='store_true', 
                    help='whether restore from previous trained model')
        parser.add_argument('--use_crf', action='store_false', 
                    help='whether use crf optimization layer')
        parser.add_argument('--train_embeddings', action='store_true', 
                    help='whether use cnn or lstm for sentence representation')
        parser.add_argument('--use_pretrained', action='store_false', 
                    help='whether use pre-trained word embeddings')
        parser.add_argument('--train_accuracy', action='store_false', 
                    help='whether report accuracy while training')

        self.parser.parse_args(namespace=self)

        self.dir_output = os.path.join('results', self.dir_output)
        self.dir_model  = os.path.join(self.dir_output, "model.weights")
        self.path_log   = os.path.join(self.dir_output, "log.txt")

        # dataset
        self.filename_dev = os.path.join(self.data_root, 'PICO_dev.txt')
        self.filename_test = os.path.join(self.data_root, 'PICO_test.txt')
        self.filename_train = os.path.join(self.data_root, 'PICO_train.txt')

        # vocab (created from dataset with build_data.py)
        self.filename_words = os.path.join('data', 'words.txt')
        self.filename_tags = os.path.join('data', 'tags.txt')

        # directory for training outputs
        if not os.path.exists('data'):
            os.makedirs('data')

        # directory for data output
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)

        # create instance of logger
        self.logger = get_logger(self.path_log)

        # log the attributes
        msg = ', '.join(['{}: {}'.format(attr, getattr(self, attr)) for attr in dir(self) \
                        if not callable(getattr(self, attr)) and not attr.startswith("__")])
        self.logger.info(msg)

        # load if requested (default)
        if load:
            self.load()


    def load(self):
        """Loads vocabulary, processing functions and embeddings

        Supposes that build_data.py has been run successfully and that
        the corresponding files have been created (vocab and trimmed GloVe
        vectors)

        """
        # 1. vocabulary
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags  = load_vocab(self.filename_tags)

        self.nwords     = len(self.vocab_words)
        self.ntags      = len(self.vocab_tags)

        # 2. get processing functions that map str -> id
        self.processing_word = get_processing_word(self.vocab_words, lowercase=True)
        self.processing_tag  = get_processing_word(self.vocab_tags,
                lowercase=False, allow_unk=False)

        # 3. get pre-trained embeddings
        self.embeddings = (get_trimmed_wordvec_vectors(self.filename_wordvec_trimmed, self.vocab_words)
                if self.use_pretrained else None)
        self.dim_word = self.embeddings.shape[1]


    max_iter = None # if not None, max number of examples in Dataset

