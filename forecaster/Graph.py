
import pickle
import matplotlib.pyplot as plt
import sys

f = open (sys.argv[1], "rb")
history_dict = pickle.load (f)

loss_values = history_dict ['loss']
val_loss_values = history_dict['val_loss']
    
    

epochs = range (0, len(loss_values))
plt.plot (epochs, loss_values, 'bo', label='Training Loss')
plt.plot (epochs, val_loss_values, 'b', label='Validation Loss')
plt.title ('Training and validation loss')
plt.xlabel ('Epochs')
plt.ylabel ('Loss')
plt.legend ()
        
plt.show()
