import warnings
import os

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)

class Logger:
    def __init__(self, file_path, clear=False) -> None:
        self.file_path = file_path
        if clear:
            self._clean_history()
    
    def _clean_history(self):
        with open(self.file_path, 'w') as f:
            f.write('')
        print('Log history has been cleaned.')        
    
        
    def fprint(self, msg):
        with open(self.file_path, 'a+') as f:
            f.write(msg)
            f.write('\n')
        print(msg)
    
    def warn(self, msg, warn_type):
        with open(self.file_path, 'a+') as f:
            f.write(f"{str(warn_type)}: {msg}")
            f.write('\n')
        warnings.warn(msg, warn_type)
        
        
if __name__ == "__main__":
    logger = Logger('test.log', True)
    logger.fprint("Running exps")
    logger.warn('Syntax warning', SyntaxWarning)