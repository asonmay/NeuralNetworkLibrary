using System;
using System.Collections.Generic;
using System.Linq;
using System.Net.Http.Headers;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;
using System.Transactions;

namespace NeuralNetworkLibrary
{
    public class Perceptron
    {
        public double[] Weights { get; private set; }
        public double Bias { get; private set; }
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

        private (double[], double) MutateHillClimber()
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
            (double[], double) mutated = MutateHillClimber();
            double mutatedError = GetError(inputs, desiredOutputs, mutated.Item1, mutated.Item2);

            if(mutatedError < currentError || currentError == 0)
            {
                Weights = mutated.Item1;
                Bias = mutated.Item2;
                return mutatedError;
            }

            return currentError;
        }

        private double CalculateWeightPartiaDerivative(int index, double[] inputs, double desiredOutputs)
        {
            double sum = 0;
            for (int i = 0; i < inputs.Length; i++) sum += inputs[i] * Weights[i];

            return errorFunc.Derivative(Compute(inputs, Weights, Bias), desiredOutputs) + activationFunction.Derivative(sum + Bias) + inputs[index];
        }

        private double CalculateBiasPartialDerivitive(double[] inputs, double desiredOutputs)
        {
            double sum = 0;
            for (int i = 0; i < inputs.Length; i++) sum += inputs[i] * Weights[i];

            return errorFunc.Derivative(Compute(inputs, Weights, Bias), desiredOutputs) + activationFunction.Derivative(sum + Bias);
        }

        private (double[], double) GetMutateValues(double[] inputs, double desiredOutput)
        {
            double[] WeightValues = new double[Weights.Length];
            double BiasValue;

            for (int i = 0; i < Weights.Length; i++)
            {
                WeightValues[i] = weightMutationAmount * -CalculateWeightPartiaDerivative(i, inputs, desiredOutput);
            }
            BiasValue = biasMutationAmount * -CalculateBiasPartialDerivitive(inputs, desiredOutput);

            return (WeightValues, BiasValue);
        }

        private (double[], double) GetBachMutateValues(double[][] inputs, double[] desiredOutputs)
        {
            (double[], double)[] mutateValues = new (double[], double)[inputs.GetLength(1)];
            for (int i = 0; i < inputs.GetLength(1); i++)
            {
                mutateValues[i] = GetMutateValues(inputs[i], desiredOutputs[i]);
            }
            double[] weightMutateValues = new double[Weights.Length];
            double biasMutateValue = 0;

            for (int i = 0; i < inputs.GetLength(1); i++)
            {
                for (int j = 0; j < inputs.GetLength(0); j++)
                {
                    weightMutateValues[i] += mutateValues[i].Item1[j];
                }
                biasMutateValue += mutateValues[i].Item2;
            }

            return (weightMutateValues, biasMutateValue);
        }

        private (double[], double) Mutate(double[] weightMutationAmounts, double biasMutationAmount)
        {
            double[] newWeights = new double[Weights.Length];
            Weights.CopyTo(newWeights, 0);
            double newBias = Bias;

            for(int i = 0; i < Weights.Length;i++)
            {
                newWeights[i] += weightMutationAmounts[i];
            }
            newBias += biasMutationAmount;

            return (newWeights, newBias);
        }

        public double TrainWithGradientDescent(double[][] inputs, double[] desiredOutputs, double currentError)
        {
            (double[], double) mutationAmounts = GetBachMutateValues(inputs, desiredOutputs);
            (double[], double) mutated = Mutate(mutationAmounts.Item1, mutationAmounts.Item2);

            double mutatedError = GetError(inputs, desiredOutputs, mutated.Item1, mutated.Item2);

            if (mutatedError < currentError || currentError == 0)
            {
                Weights = mutated.Item1;
                Bias = mutated.Item2;
                return mutatedError;
            }

            return currentError;
        }
    }
}
