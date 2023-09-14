import warnings

class Logger:
    def __init__(self, file_path, clear=False) -> None:
        self.file_path = file_path
        if clear:
            self._clean_history()
    
    def _clean_history(self):
        with open(self.file_path, 'w') as f:
            f.write('')
        print('Log history has been cleaned.')        
    
        
    def fprint(self, text):
        with open(self.file_path, 'a+') as f:
            f.write(text)
            f.write('\n')
        print(text)
    
    def warn(self, text, warn_type):
        with open(self.file_path, 'a+') as f:
            f.write(f"{str(warn_type)}: {text}")
            f.write('\n')
        warnings.warn(text, warn_type)
        
        
if __name__ == "__main__":
    logger = Logger('test.log', True)
    logger.fprint("Running exps")
    logger.warn('Syntax warning', SyntaxWarning)