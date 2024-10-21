# featureCsignal
## Explainability for simple feed forward neural networks on Tensorflow
This package provides the code to analyze your simple neural network and explain the results starting from the input features. 

Using a simple trick it is possible to isolate each contribution of the input features over the particular output.

### Explainability
The results will show how much signal each feature passes to the output. Summing up all the signals gives the output value.

The interesting result is that you may have features that reduces the output value or features that, while seemingly being strong, provide little value at the end. To get the most out of your understanding what your model has learned and why the result is like it is, take care of interpreting in a <b>conditional way</b>. In fact, each signal passes through the non linearities together with the others, meaning that the model may have learned to give importance to some features under some circumnstances or providing that the other values are what they are in the moment. 

To better understand the conditionality, think that by changing a single input feature value, you obtain a change even on the other features output values. Meaning that, in a neural network contest, features are not independent and cannot be interpreted as independent. 


### Support of package
This is a very experimental project and works at its full with simple neural networks. It supports:
<ul><li>Dense layer</li>
<li>Reshape layer</li>
<li>Flatten layer</li>
<li>Batchnorm layer</li>
<li>Conv1d layer</li></ul>
and single outputs.

With multiple outputs the results have to be analyzed manually and without the methods provided. 

The package, in this moment, <b>does not support</b> more complex neural networks like networks with multiple branches (i.e. skip connections).

A little warning is that by using Batchnorm and Conv1d layers the final output is a little different due to numerical instability in calculations. 

### Support of project
If you want to use the code, or the idea under the code, please see the LICENCE. 

If you want to contribute to the project, please contact me (or open an issue).


Thanks for being here, RV.


