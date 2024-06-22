
from ShakespeareanGenerator.logger import Logger
from ShakespeareanGenerator.artifact_manager import ObjectStoreArtifactManager

class Run:
    def __init__(self):
        self.logging = Logger()
        self.obj = ObjectStoreArtifactManager()
        self.prepare_data()

    def prepare_data(self):
        self.logging.info('START: PREPARATION STEP')
        self.obj.upload_file_to_object_store()
        self.logging.info('Training Data was uploaded to Object Store')
        self.obj.copy_object(model_type='model')
        self.logging.info('The Language Model was successfully uploaded to the object store')
        self.obj.copy_object(model_type='bpe_model')
        self.logging.info('The trained tokenizer (BPE) was successfully uploaded to the object store')
        self.logging.info('END: PREPARATION STEP')

if __name__ == '__main__':
    Run()