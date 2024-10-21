import numpy as np 
import pandas as pd 
from tensorflow import keras
import tensorflow as tf
from scipy.stats import t


class InputCSignal:
    """
    Create a class for analyzing each feature conditional signal passed over the
    network to the output. 
    Useful to explain decision made by the nn, assert importance of fetaures and 
    select them using statistical tools.
    """
    
    def __init__(self):
        self.model = None

    def fit(self, modell):
        """Fit the model trained into a class copy"""
        self.model = keras.models.clone_model(modell)
        self.model.build((None, modell.layers[0].weights[0].shape[0])) 
        self.model.compile(optimizer=modell.optimizer, loss=modell.loss)
        self.model.set_weights(modell.get_weights())

    def transform(self, example):
        """
        Calculate the conditional signal linked to each feature in the single example.
        The usage of BatchNormalization and Conv1D layers gives results that differ from the predict because of calculation instability.
        More complex nn having multiple feature layers i.e. skip connections are not supported.
        """
        inp, dim = example, example.shape[1]
        inp = np.eye(dim)*inp

        for layer in self.model.layers:
            #print(layer.__class__.__name__)
            if layer.__class__.__name__ == "BatchNormalization":
                total_effect = np.sum(inp, axis=0)
                activation = (np.sum(inp, axis=0) - layer.get_weights()[2])/np.sqrt(layer.get_weights()[3])*layer.get_weights()[0]+layer.get_weights()[1]
                inp = inp/(total_effect+1e-16)*activation # spread equally the effect
                inp[:, total_effect == 0] = activation[total_effect == 0]/dim
            if layer.__class__.__name__ == "Reshape":
                sha = tf.TensorShape(layer.output.shape).as_list()
                sha[0] = -1
                inp = np.reshape(inp, sha)
            if layer.__class__.__name__ == "Flatten":
                sha = tf.TensorShape(layer.output.shape).as_list()
                sha[0] = -1
                inp = np.reshape(inp, sha)
            if layer.__class__.__name__ == "Dense":
                inp = inp.dot(layer.get_weights()[0]) + layer.get_weights()[1]/dim
                total_effect = np.sum(inp, axis=0)
                activation = layer.activation(total_effect)
                inp = inp/(total_effect+1e-16)*activation
            if layer.__class__.__name__ == "Conv1D":
                inp = np.array([np.convolve(i.squeeze(), layer.get_weights()[0].squeeze(), layer.padding) + layer.get_weights()[1]/dim for i in inp])
                total_effect = np.sum(inp, axis=0)
                activation = layer.activation(total_effect)
                inp = inp/(total_effect+1e-16)*activation

            inp = np.array(inp)
        return inp.squeeze()

    def fit_transform(self, modell, example):
        """Fit model and transform single example"""
        self.fit(modell)
        return self.transform(example)

    def fit_transform_all(self, modell, examples):
        """Fit model and transform all examples"""
        self.fit(modell)
        return self.transform_all(examples)

    def transform_all(self, examples):
        """Transform all examples"""
        allex = []
        for i in range(len(examples)):
            allex.append(self.transform(examples[i:i+1]))
        return np.array(allex).squeeze()

    def transform_summary(self, example, columns=None, w_total = True):
        """Transform single example and give a summary of the signals and their relative importance in the example"""
        x = self.transform(example)
        if (columns == None).all():
            columns = [f'var {i}' for i in range(len(example))]
        df = pd.DataFrame([columns, example[0], x, x*100/np.sum(x)], index=["var", "value", "signal", "relative importance (on 100)"]).T
        if w_total:
            df.loc[len(df.index)] = ['total', None, np.sum(x), np.sum(x*100/np.sum(x))] 
        return df

    def fit_transform_summary(self, modell, example, columns=None, w_total=True):
        """Fit model and transform single example with summary and relative importance"""
        self.fit(modell)
        return self.transform_summary(example, columns, w_total)

    def fit_transform_all_summary(self, modell, examples, columns=None, correction=False):
        """Fit model and transform all examples with summary over the whole signals and their total significance level - one output only"""
        self.fit(modell)
        return self.transform_all_summary(examples, columns, correction)

    def transform_all_summary(self, examples, columns=None, correction=False):
        """Transform all examples and make summary of the whole feature's signals and their significance - one output only"""
        allex = self.transform_all(examples)
        if correction:
            ttest = np.mean(allex, axis=0)/np.std(allex, axis=0)*np.sqrt(len(allex)/len(np.mean(allex, axis=0)))
        else:
            ttest = np.mean(allex, axis=0)/np.std(allex, axis=0)*np.sqrt(len(allex))
        pval = t.pdf(np.abs(ttest), df=len(allex)-len(np.mean(allex, axis=0))-1)
        sign = [self.__sign__(np.mean(i)) for i in pval]
        if (columns == None).all():
            columns = [f'var {i}' for i in range(len(example))]
        return pd.DataFrame([columns, np.mean(allex, axis=0), np.std(allex, axis=0), 
                    ttest, pval, sign],
                           index=["var", "mean", "std", "t_test", "p-value", "significance"]).T

    def __sign__(self, i): # utility function for significance
        s = ""
        if (i > 0.05).all():
            s = " "
        if ((i<=0.05)&(i>0.025)).all():
            s = "*"
        if ((i<=0.025)&(i>0.001)).all():
            s = "**"
        if (i<=0.001).all():
            s = "***"
        return s
