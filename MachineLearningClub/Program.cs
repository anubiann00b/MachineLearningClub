using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// COME TO MACHINE LEARNING CLUB IN O'BYRNE/THOMPSON ROOM ON
// TUESDAYS AND THURSDAYS.

namespace MachineLearningClub
{
    /**
    * Simple example of a basic perceptron.
    */
    class PerceptronTester
    {
        static void Main(string[] args)
        {
            // Test cases (that are really bad).
            double[][] input = new double[][]
            {
                // Each array is a test case with 'length' dimensions
                new double[]{0.0},
                new double[]{0.1},
                new double[]{0.2},
                new double[]{0.3},
                new double[]{0.4},
                new double[]{0.5},
                new double[]{0.6},
                new double[]{0.7},
                new double[]{0.8},
                new double[]{0.9},
                new double[]{1.0},
            };

            // The expected output for each test case.
            double[] output = new double[] { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
            
            NeuralNetwork network = new NeuralNetwork(input[0].Length, 5);
            NetworkTrainer teacher = new NetworkTrainer(network, 0.05D);

            // Run the training 1000 times, since we don't have enough test cases.
            for (int x = 0; x < 1000; x++)
            {
                for (int i = 0; i < output.Length; i++)
                {
                    double error = teacher.Teach(input[i], output[i]);
                    Console.WriteLine(Math.Round(error, 3) + "\t" + string.Join(",",network.HiddenOut));
                }
                Console.WriteLine();
            }
            
        }
    }

    public class NeuralNetwork
    {
        public Neuron[] Hidden;
        public Neuron Output;

        public double[] HiddenOut;

        private Random random = new Random();

        public NeuralNetwork(int numIn, int numHidden)
        {
            HiddenOut = new double[numHidden];
            this.Hidden = Enumerable.Range(0, numHidden)
                .Select(x => new Neuron(numIn))
                .ToArray();
            this.Output = new Neuron(numHidden);
        }

        public double Compute(double[] input)
        {
            for (int i = 0; i < Hidden.Length; i++)
            {
                HiddenOut[i] = Hidden[i].Run(input);
            }

            return Output.Run(HiddenOut);
        }
    }

    public class Neuron
    {
        // The number of inputs (dimension).
        public readonly int Inputs;

        // The weight for each input, from -1 to 1.
        private double[] weights;

        // Allows us to do C# magic, like PerceptronNetwork[i].
        public double this[int index]
        {
            get { return this.weights[index]; }
            set { this.weights[index] = value; }
        }

        private Random random = new Random();

        public Neuron(int inputs)
        {
            this.Inputs = inputs;

            // Get an array of random values from -1 to 1.
            this.weights = Enumerable.Range(0, this.Inputs)
                .Select(x => random.NextDouble() * 2 - 1)
                .ToArray();
        }

        public double Run(double[] inp)
        {
            // Make sure the user isn't an idiot.
            Trace.Assert(weights.Length == inp.Length);

            // Get the sum of each input multiplied by its corresponding weight.
            double sum = Enumerable
                .Range(0, weights.Length)
                .Select(x => weights[x] * inp[x])
                .Sum();

            return Sigmoid(sum);
        }

        public static double Sigmoid(double d)
        {
            return 1 / Math.Exp(-d);
        }
    }

    public class NetworkTrainer
    {
        public NeuralNetwork Network { get; private set; }

        // Constant representing training speed.
        public double Alpha { get; set; }

        public NetworkTrainer(NeuralNetwork network, double alpha)
        {
            this.Network = network;
            this.Alpha = alpha;
        }

        public double Teach(double[] input, double expectedOut)
        {
            // Keep track of the total error. Should be converging to a small value.
            double sum = 0;

            // Error = Expected - Actual
            double error = expectedOut - this.Network.Compute(input);

            // Based on the error set new weights.
            for (int i = 0; i < this.Network.Hidden.Length; i++)
            {
                for (int j = 0; j < this.Network.Hidden[i].Inputs; j++)
			    {
                    // Formula: new weight = old weight * alpha * input value
                    this.Network.Hidden[i][j] += this.Alpha * error * input[j];
                    sum += error;
			    }
            }

            for (int i = 0; i < this.Network.Output.Inputs; i++)
            {
                this.Network.Output[i] += this.Alpha * error * Network.HiddenOut[i];
            }

            return sum;
        }
    }
}
