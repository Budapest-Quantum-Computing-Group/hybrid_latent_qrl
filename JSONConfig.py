import json

class JSONConfig():
    def __init__(self):
        self.keys = {}
        self._typefn = type
    
    def add_key(self, name, type, required=False, choices=None, default=None):
        
        if choices is not None and not all([self._typefn(c) == type for c in choices]):
            raise ValueError(f"All values in 'choices' must be {type}")
            
        if default is not None:
            if self._typefn(default) != type:
                raise ValueError(f"Expected default value to be {type} but is {self._typefn(default)}")
            
        if choices is not None and default is not None and default not in choices:
                raise ValueError(f"Expected default value to be one of {choices}, got {default}")
            
        self.keys[name] = {'type': type, 'required': required, 'choices': choices, 'default': default}
        self.update({name: default})
        
        
    def all_required_keys_present(self, d):
        # Check existence of required keys
        for k in self.keys.keys():
            if self.keys[k]['required'] == True:
                if not k in d.keys():
                    raise ValueError(f"Required key {k} not found.")
                if k in d.keys() and d[k] is None:
                    raise ValueError(f"Required key {k} not found.")
        return True
    
    def is_valid(self, d): 
        # Check if the type of the given keys are correct
        for k in d.keys():
            if k not in self.keys.keys():
                raise ValueError(f"Unrecognized key {k}")
                
            if d[k] and d[k] is not None:
                if self._typefn(d[k]) != self.keys[k]['type']:
                    raise ValueError(f"Bad type for key {k}. Expected {self.keys[k]['type']} got {self._typefn(d[k])}")
                
            else:
                if self.keys[k]['choices'] is not None:
                    if d[k] and d[k] not in self.keys[k]['choices']:
                        raise ValueError(f"Expected one of {self.keys[k]['choices']}, got {d[k]}")
                
        return True
    
    def __dict_no_keys(self):
        d = self.__dict__.copy()
        d.__delitem__('keys')
        return d
    
    def __str__(self):
        return str(self.__dict_no_keys())

    def print(self):
        print(self.__dict_no_keys())
        
    def update(self, d):
        if self.is_valid(d):
            self.__dict__.update(d)
            
    def load(self, file):
        with open(file, 'r') as f:
            d = json.loads(f.read())
            f.close()
            
        if self.all_required_keys_present(d) and self.is_valid(d):
            self.__dict__.update(d)
            
    def save(self, file):
        with open(file, 'w') as f:
            json.dump(self.__dict_no_keys(), f, indent=4, sort_keys=True)
        f.close()