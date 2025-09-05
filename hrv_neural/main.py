import argparse
from code import mapping, cost, evolution, representation, manifold, regularization, phantasy

def main():
    parser = argparse.ArgumentParser(description="HRV-Centric State Transmission Demo")
    parser.add_argument('--module', choices=['mapping','cost','evolution','representation','manifold','regularization','phantasy'], default='mapping')
    args = parser.parse_args()

    if args.module == 'mapping':
        print(mapping.mapping_function())
    elif args.module == 'cost':
        print(cost.demo_loss())
    elif args.module == 'evolution':
        evolution.demo()
    elif args.module == 'representation':
        representation.demo()
    elif args.module == 'manifold':
        manifold.demo()
    elif args.module == 'regularization':
        regularization.demo()
    elif args.module == 'phantasy':
        phantasy.demo()

if __name__ == "__main__":
    main()
