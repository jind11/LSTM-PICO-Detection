from model.data_utils import Dataset
from model.models import HANNModel
from model.config import Config
import argparse

parser = argparse.ArgumentParser()

def main():
    # create instance of config
    config = Config(parser)

    # build model
    model = HANNModel(config)
    model.build()
    if config.restore:
        model.restore_session("results/test/model.weights/") # optional, restore weights
    # model.reinitialize_weights("proj")

    # create datasets
    dev   = Dataset(config.filename_dev, config.processing_word,
                         config.processing_tag, config.max_iter)
    train = Dataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)
    test  = Dataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)

    # train model
    model.train(train, dev)

    # evaluate model
    model.evaluate(test)

if __name__ == "__main__":
    main()
