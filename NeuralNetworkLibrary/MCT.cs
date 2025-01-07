using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetworkLibrary
{
    public class MCT<T> where T : IGameState<T>
    {
        public Node<T> Head { get; }
        private double c;
        private int iterations;
        private Random random;

        public MCT(double c, int iterations) 
        {
            this.c = c;
            this.iterations = iterations;
            random = new Random();
        }

        public MCT(Node<T> head, double c, int iterations)
        {
            Head = head;
            this.c = c;
            this.iterations = iterations;
            random = new Random();
        }

        public Node<T> Select(Node<T> root)
        {
            Node<T> currentNode = root;
            while(currentNode.IsExpanded)
            {
                Node<T> highestUCTNode = null;
                double highestUCT = double.NegativeInfinity;

                for(int i = 0; i < currentNode.Children.Length; i++)
                {
                    double val = currentNode.Children[i].UCT(c);
                    if (val > highestUCT)
                    {
                        highestUCTNode = currentNode.Children[i];
                        highestUCT = val;
                    }
                }

                if(highestUCTNode == null)
                {
                    break;
                }
                currentNode = highestUCTNode;
            }
            return currentNode;
        }

        public Node<T> Expand(Node<T> currentNode)
        {
            currentNode.GenerateChildren();

            if (currentNode.Children.Length == 0)
            {
                return currentNode;
            }
            else
            {
                return currentNode.Children[0];
            }
        }

        public int Simulate(Node<T> startingNode)
        {
            Node<T> currentNode = startingNode;
            while(!currentNode.GameState.IsTerminal)
            {
                currentNode.IsExpanded = true;
                currentNode.GenerateChildren();
                int randomIndex = random.Next(0, currentNode.Children.Length);
                currentNode = currentNode.Children[randomIndex];
            }

            if (currentNode.GameState.IsWin)
            {
                return 1;
            }
            else if (currentNode.GameState.IsTie)
            {
                return 0;
            }
            else
            {
                return -1;
            }
        }

        public void Backpropagate(int value, Node<T> currentNode)
        {
            while (currentNode != null)
            {
                value = -value;
                currentNode.N++;
                currentNode.W += value;
                currentNode = currentNode.Parent;
            }
        }

        public void GenerateTree(Node<T> startingNode, int iterations)
        {
            for (int i = 0; i < iterations; i++)
            {
                Node<T> selectedNode = Select(startingNode);
                Node<T> expandedChild = Expand(selectedNode);
                int value = Simulate(expandedChild);
                Backpropagate(value, expandedChild);
            }
        }

        public Node<T> GetBestMove(Node<T> currentNode)
        {
            GenerateTree(currentNode, iterations);

            var sortedChildren = currentNode.Children.OrderByDescending((state) => state.W);
            var topChild = sortedChildren.First();
            return topChild;
        }
    }
}
