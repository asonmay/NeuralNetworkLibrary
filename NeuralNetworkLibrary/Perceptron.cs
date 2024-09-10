using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class Perceptron
    {
        public double[] Weights;
        public double Bias;
        private double weightMutationAmount;
        private double biasMutationAmount;
        private Random random;
        private ActivationFunction activationFunction;
        private ErrorFunction errorFunc;

        public Perceptron(double[] initialWeightValues, double initialBiasValue,
          double weightMutationAmount, double biasMutationAmount, ErrorFunction errorFunc, ActivationFunction activationFunction)
        {
            Weights = initialWeightValues;
            Bias = initialBiasValue;
            this.weightMutationAmount = weightMutationAmount;
            this.biasMutationAmount = biasMutationAmount;
            random = new Random();
            this.errorFunc = errorFunc;
            this.activationFunction = activationFunction;
        }

        public void Randomize(int min, int max)
        {
            for(int i = 0; i < Weights.Length; i++) Weights[i] = random.Next(min, max);
        }

        private (double[], double) Mutate()
        {
            double[] newWeights = new double[Weights.Length];
            Weights.CopyTo(newWeights, 0);
            double newBias = Bias;

            if(random.Next(0,2) == 0)
            {
                newWeights[random.Next(Weights.Length)] += random.Next(0, 2) == 0 ? weightMutationAmount : -weightMutationAmount;
            }
            else
            {
                newBias += random.Next(0, 2) == 0 ? biasMutationAmount : -biasMutationAmount;
            }

            return (newWeights, newBias);
        }
        
        public double Compute(double[] inputs, double[] weight, double bias)
        {
            double sum = 0;
            for (int i = 0; i < inputs.Length; i++) sum += inputs[i] * weight[i];

            return activationFunction.Function(sum + bias);
        }

        public double[] Compute(double[][] inputs, double[] weights, double bias)
        {
            double[] result = new double[inputs.GetLength(0)];
            for (int i = 0; i < inputs.Length; i++)
            {
                result[i] = Compute(inputs[i], weights, bias); 
            }
            return result;
        }

        public double GetError(double[][] inputs, double[] desiredOutputs, double[] weights, double bias)
        {
            double error = 0;
            double[] values = Compute(inputs, weights, bias);
            for(int i = 0; i < values.Length; i++)
            {
                error += errorFunc.Function(values[i], desiredOutputs[i]);
            }            

            return error / values.Length;
        }

        public double TrainWithHillClimbing(double[][] inputs, double[] desiredOutputs, double currentError)
        {
            (double[], double) mutated = Mutate();
            double mutatedError = GetError(inputs, desiredOutputs, mutated.Item1, mutated.Item2);

            if(mutatedError < currentError || currentError == 0)
            {
                Weights = mutated.Item1;
                Bias = mutated.Item2;
                return mutatedError;
            }

            return currentError;
        }
    }
}
