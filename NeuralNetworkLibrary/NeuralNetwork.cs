using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection.Metadata.Ecma335;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class NeuralNetwork
    {
        public Layer[] Layers;
        private ErrorFunction errorFunc;
        private ActivationFunction activation;

        public NeuralNetwork(ActivationFunction activation, ErrorFunction errorFunc, params int[] neuronsPerLayer)
        {
            Layers = new Layer[neuronsPerLayer.Length];
            Layers[0] = new Layer(activation, neuronsPerLayer[0], null);
            for (int i = 1; i < neuronsPerLayer.Length; i++)
            {
                Layers[i] = new Layer(activation, neuronsPerLayer[i], Layers[i-1]);
            }
            this.errorFunc = errorFunc;
            this.activation = activation;
        }

        public void Randomize(Random random, double min, double max) 
        {
            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].Randomize(random, min, max);
            }
        }

        public void ApplyChanges()
        {
            for(int i = 0; i < Layers.Length; i++)
            {
                Layers[i].ApplyChanges();
            }
        }

        public double[] Compute(double[] inputs) 
        {
            Layers[0].SetOutputs(inputs);

            for(int i = 1; i < Layers.Length; i++)
            {
                Layers[i].Compute();
            }

            return Layers[Layers.Length - 1].Outputs;
        }

        public double[] ApplyActivation(double[] inputs)
        {
            double[] newValues = new double[inputs.Length];
            for(int i = 0; i < newValues.Length; i++)
            {
                newValues[i] = activation.Function(inputs[i]);
            }
            return newValues;
        }

        public void BackProp(double learningRate, double[] desiredOutput, double[] inputs)
        {
            for(int i = 0; i < Layers[Layers.Length - 1].Neurons.Length; i++)
            {
                Layers[Layers.Length - 1].Neurons[i].Delta = errorFunc.Derivative(Layers[Layers.Length - 1].Neurons[i].Output, desiredOutput[i]);
            }
            
            for(int i = Layers.Length - 1; i > 0; i--)
            {
                Layers[i].BackProp(learningRate);
            }
        }

        public double GetError(double[] inputs, double[] desiredOutputs) 
        {
            double[] output = Compute(inputs);
            double[] error = new double[output.Length];
            for(int i = 0; i < output.Length; i++)
            {
                error[i] = errorFunc.Function(output[i], desiredOutputs[i]);
            }

            return error.Sum();
        }

        public double GetError(double[][] inputs, double[][] desiredOutputs)
        {
            double error = 0;
            for(int i = 0; i < inputs.Length; i++)
            {
                error += GetError(inputs[i], desiredOutputs[i]);
            }
            return error / inputs.Length;
        }
    }
}
