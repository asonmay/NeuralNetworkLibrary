using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class GradientDescentTrainer
    {
        public NeuralNetwork Network;
        private double[][] inputs;
        private double[][] desiredOutputs;

        public GradientDescentTrainer(double[][] inputs, double[][]desiredOutputs, ActivationFunction activation, ErrorFunction error, int min, int max, params int[] neuronsPerLayer)
        {
            this.inputs = inputs;
            this.desiredOutputs = desiredOutputs;
            Network = new NeuralNetwork(activation, error, neuronsPerLayer);
            Network.Randomize(new Random(), min, max);
        }

        public double Train(double learningRate)
        {
            for(int i = 0; i < desiredOutputs.Length; i++)
            {
                Network.BackProp(learningRate, desiredOutputs[i]);
            }
            Network.ApplyChanges();

            double error = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                error = Network.GetError(inputs[i], desiredOutputs[i]);
            }
            return error / desiredOutputs.Length;
        }

        public void Train(double learningRate, double iterations)
        {
            double current = 0;
            while(current < iterations)
            {
                Train(learningRate);
                current++;
            }
        }
    }
}
