using System;
using System.Collections.Generic;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class GradientDescentTrainer
    {
        public NeuralNetwork Network;
        private double[][] inputs;
        private double[][] desiredOutputs;

        public GradientDescentTrainer(double[][] inputs, double[][]desiredOutputs, ActivationFunction activation, ErrorFunction error, double min, double max, params int[] neuronsPerLayer)
        {
            this.inputs = inputs;
            this.desiredOutputs = desiredOutputs;
            Network = new NeuralNetwork(activation, error, neuronsPerLayer);
            Network.Randomize(new Random(), min, max);
        }

        public double Train(double learningRate, double currentError)
        {
            for(int i = 0; i < desiredOutputs.Length; i++)
            {
                Network.Compute(inputs[i]);
                Network.BackProp(GetLearningRate(currentError, learningRate), desiredOutputs[i], inputs[i]);
            }
            Network.ApplyChanges();

            double error = 0;
            for (int i = 0; i < inputs.Length; i++)
            {
                error += Network.GetError(inputs[i], desiredOutputs[i]);
            }
            error = error / inputs.Length;
            Console.SetCursorPosition(0, 0);
            Console.Write(error);
            return error;
        }

        public void Train(double learningRate, int iterations, double currentError)
        {
            double current = 0;
            while(current < iterations)
            {
                Train(learningRate, currentError);
                current++;
            }
        }

        private double GetLearningRate(double error, double original)
        {
            return original * error;
        }

        public void Train(double learningRate, float error)
        {
            double currentError = Network.GetError(inputs, desiredOutputs);
            while (currentError >= error)
            {
                currentError = Train(learningRate, currentError);
                for(int i = 0; i < inputs.Length; i++)
                {
                    Console.SetCursorPosition(0, i + 1);
                    //Console.Write("the sine of ");
                    //Console.Write(inputs[i][0]);
                    //Console.Write(" is ");
                    //Console.Write(Network.Compute(inputs[i])[0]);
                    Console.Write(inputs[i][0]);
                    Console.Write(" XOR ");
                    Console.Write(inputs[i][1]);
                    Console.Write(" is ");
                    Console.Write(Network.Compute(inputs[i])[0]);
                }
            }
        }
    }
}
