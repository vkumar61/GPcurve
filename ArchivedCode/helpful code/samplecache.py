
import h5py
import numpy as np
from types import SimpleNamespace

# Sample cache class for saving MCMC samples
class SampleCache:

    def __init__(self, save_name, variables=None, fields=None, num_iter=None, path=None):

        # Check extension
        if not save_name.lower().endswith('.h5'):
            save_name += '.h5'

        # Check path
        if path is None:
            path = ''
        elif path[-1] != '/':
            path += '/'

        # Set fields
        if fields is None:
            fields = []
        self.fields = fields

        # Set attributes
        self.save_name = save_name
        self.path = path
        self.fields = fields
        self.num_iter = num_iter

        # Initialize
        if variables is not None:
            self.initialize(variables)
        else:
            self.load_history()

        return

    def load_history(self):

        # Get attributes
        save_name = self.save_name
        path = self.path

        # Get fields
        h5 = h5py.File(path + save_name, 'r')
        fields = list(h5.keys())
        h5.close()
        fields = [field for field in fields if field.lower() != 'map']

        # Get num_iter
        if len(fields) > 0:
            h5 = h5py.File(path + save_name, 'r')
            num_iter = h5[fields[0]].shape[0]
            h5.close()
        else:
            num_iter = None

        # Set attributes
        self.fields = fields
        self.num_iter = num_iter

        return

    def initialize(self, variables):

        # Get attributes
        save_name = self.save_name
        path = self.path
        fields = self.fields
        num_iter = self.num_iter

        # Check if num_iter is specified
        if num_iter is None:
            raise Exception('num_iter must be specified when saving')

        # Open h5
        h5 = h5py.File(path + save_name, 'w')

        # Initialize fields
        for field in fields:
            variables_field = getattr(variables, field)
            chunk = (1, *np.shape(variables_field))
            if len(chunk) == 1:
                chunk = (1, 1)
                dtype = type(variables_field)
            else:
                dtype = variables_field.dtype.type
            shape = (num_iter, *chunk[1:])
            h5.create_dataset(name=field, shape=shape, chunks=chunk, dtype=dtype)

        # Initialize MAP
        h5.create_group('MAP')
        for key, value in variables.__dict__.items():
            try:
                h5['MAP'].create_dataset(name=key, data=value)
            except:
                h5['MAP'].create_dataset(name=key, data=repr(value))
        
        h5.close()

        return

    def checkpoint(self, variables, iter_num):

        # Get attributes
        save_name = self.save_name
        path = self.path
        fields = self.fields

        # Check if num_iter is specified
        if iter_num == 0:
            self.initialize(variables)
        
        # Open h5 file
        h5 = h5py.File(path + save_name, 'r+')

        # Save MAP
        if variables.P >= np.max([-np.inf, *self.get('P')[:iter_num, 0]]):
            del h5['MAP']
            h5.create_group('MAP')
            for key, value in variables.__dict__.items():
                try:
                    h5['MAP'].create_dataset(name=key, data=value)
                except:
                    h5['MAP'].create_dataset(name=key, data=repr(value))

        # Save fields
        for field in fields:
            h5[field][iter_num, :] = getattr(variables, field)

        # close h5 file
        h5.close()

        return

    def get(self, field, burn=0, last=None):

        # Get attributes
        save_name = self.save_name
        fields = self.fields
        path = self.path

        # Open h5 file
        h5 = h5py.File(path + save_name, 'r')

        if field.lower() == 'fid':
            # Return file
            return h5
        elif field.lower() == 'map':
            # Return MAP
            value = {key:h5['MAP'][key][()] for key in h5['MAP'].keys()}
            value = SimpleNamespace(**value)
        else:
            # Return sample trace
            value = h5[field]
            if 0 < burn < 1:
                burn = round(burn * value.shape[0])
            value = value[burn:last, :]
        
        # Close h5 file
        h5.close()

        return value


if __name__ == '__main__':

    ### Test SampleCache ###

    # Initialize variables
    variables = SimpleNamespace(P=np.inf, a=1, b=2)

    # Initialize SampleCache
    savename = 'test'
    num_iter = 10
    samplecache = SampleCache(savename, variables=variables, num_iter=num_iter, fields=['P', 'a', 'b'])

    # Fake MCMC
    for i in range(10):
        variables.P = - variables.a ** 2 + variables.b ** 2
        variables.a = np.random.rand()
        variables.b = np.random.rand()
        samplecache.checkpoint(variables, i)
    
    # Get samples
    P_samples = samplecache.get('P')
    a_samples = samplecache.get('a', burn=0.5)
    b_samples = samplecache.get('b', burn=0.5)
    MAP = samplecache.get('MAP')

    print(f"a samples = {a_samples}")
    print(f"b samples = {b_samples}")
    print(f"Probabilities = {P_samples}")
    print(f"MAP = {MAP}")
    print("Done!")






