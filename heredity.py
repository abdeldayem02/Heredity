import csv
import itertools
import sys

PROBS = {

    # Unconditional probabilities for having gene
    "gene": {
        2: 0.01,
        1: 0.03,
        0: 0.96
    },

    "trait": {

        # Probability of trait given two copies of gene
        2: {
            True: 0.65,
            False: 0.35
        },

        # Probability of trait given one copy of gene
        1: {
            True: 0.56,
            False: 0.44
        },

        # Probability of trait given no gene
        0: {
            True: 0.01,
            False: 0.99
        }
    },

    # Mutation probability
    "mutation": 0.01
}


def main():

    # Check for proper usage
    if len(sys.argv) != 2:
        sys.exit("Usage: python heredity.py data.csv")
    people = load_data(sys.argv[1])

    # Keep track of gene and trait probabilities for each person
    probabilities = {
        person: {
            "gene": {
                2: 0,
                1: 0,
                0: 0
            },
            "trait": {
                True: 0,
                False: 0
            }
        }
        for person in people
    }

    # Loop over all sets of people who might have the trait
    names = set(people)
    for have_trait in powerset(names):

        # Check if current set of people violates known information
        fails_evidence = any(
            (people[person]["trait"] is not None and
             people[person]["trait"] != (person in have_trait))
            for person in names
        )
        if fails_evidence:
            continue

        # Loop over all sets of people who might have the gene
        for one_gene in powerset(names):
            for two_genes in powerset(names - one_gene):

                # Update probabilities with new joint probability
                p = joint_probability(people, one_gene, two_genes, have_trait)
                update(probabilities, one_gene, two_genes, have_trait, p)

    # Ensure probabilities sum to 1
    normalize(probabilities)

    # Print results
    for person in people:
        print(f"{person}:")
        for field in probabilities[person]:
            print(f"  {field.capitalize()}:")
            for value in probabilities[person][field]:
                p = probabilities[person][field][value]
                print(f"    {value}: {p:.4f}")


def load_data(filename):
    """
    Load gene and trait data from a file into a dictionary.
    File assumed to be a CSV containing fields name, mother, father, trait.
    mother, father must both be blank, or both be valid names in the CSV.
    trait should be 0 or 1 if trait is known, blank otherwise.
    """
    data = dict()
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row["name"]
            data[name] = {
                "name": name,
                "mother": row["mother"] or None,
                "father": row["father"] or None,
                "trait": (True if row["trait"] == "1" else
                          False if row["trait"] == "0" else None)
            }
    return data


def powerset(s):
    """
    Return a list of all possible subsets of set s.
    """
    s = list(s)
    return [
        set(s) for s in itertools.chain.from_iterable(
            itertools.combinations(s, r) for r in range(len(s) + 1)
        )
    ]


def joint_probability(people, one_gene, two_genes, have_trait):
    """
    Compute and return a joint probability.

    The probability returned should be the probability that
        * everyone in set `one_gene` has one copy of the gene, and
        * everyone in set `two_genes` has two copies of the gene, and
        * everyone not in `one_gene` or `two_gene` does not have the gene, and
        * everyone in set `have_trait` has the trait, and
        * everyone not in set` have_trait` does not have the trait.
    """
    # Dictionary to keep track of the probabilities of each person having 0, 1, or 2 copies of the gene
    prob_gene = {}
    # Dictionary to keep track of the probabilities of each person having or not having the trait
    prob_trait = {}
    # Loop over all people in the dictionary
    for person in people:
        # Check if we know how many copies of the gene the person has
        if person in one_gene:
            prob_gene[person] = 1
        elif person in two_genes:
            prob_gene[person] = 2
        else:
            prob_gene[person] = 0
        # Check if we know if the person has the trait
        if person in have_trait:
            prob_trait[person] = True
        else:
            prob_trait[person] = False
    # Loop over all people again to compute the joint probability
    joint_prob = 1
    for person in people:
        mother = people[person]['mother']
        father = people[person]['father']
        # If the person has no parents listed in the data set, use the probability distribution PROBS["gene"]
        if mother is None and father is None:
            prob = PROBS["gene"][prob_gene[person]]
        else:
            # Calculate the probability of inheriting a gene from the mother and the father
            if prob_gene[mother] == 0:
                prob_mother =PROBS["mutation"]
            elif prob_gene[mother] == 1:
                prob_mother = 0.5
            else:
                prob_mother = 1-PROBS["mutation"]
            if prob_gene[father] == 0:
                prob_father = PROBS["mutation"]
            elif prob_gene[father] == 1:
                prob_father = 0.5
            else:
                prob_father = 1-PROBS["mutation"]
            # Calculate the probability of the person having a certain number of genes based on the parents
            if prob_gene[person] == 0:
                prob = (1-prob_mother) * (1-prob_father)
            elif prob_gene[person] == 1:
                prob = (1-prob_mother) * prob_father + prob_mother * (1-prob_father)
            else:
                prob = prob_mother * prob_father
        # Calculate the probability of the person having or not having the trait based on the number of genes
        if prob_trait[person]==True:
            prob_trait_given_gene = PROBS["trait"][prob_gene[person]][True]
        else:
            prob_trait_given_gene = PROBS["trait"][prob_gene[person]][False]
        # Multiply the joint probability by the probability of the person having the gene and the trait
        joint_prob *= prob * prob_trait_given_gene
    return joint_prob



    
                




def update(probabilities, one_gene, two_genes, have_trait, p):
    """
    Add to `probabilities` a new joint probability `p`.
    Each person should have their "gene" and "trait" distributions updated.
    Which value for each distribution is updated depends on whether
    the person is in `have_gene` and `have_trait`, respectively.
    """
    # Dictionary to keep track of the probabilities of each person having 0, 1, or 2 copies of the gene
    prob_gene = {}
    # Dictionary to keep track of the probabilities of each person having or not having the trait
    prob_trait = {}
    # Loop over all people in the dictionary
    for person in probabilities:
        # Check if we know how many copies of the gene the person has
        if person in one_gene:
            prob_gene[person] = 1
        elif person in two_genes:
            prob_gene[person] = 2
        else:
            prob_gene[person] = 0
        # Check if we know if the person has the trait
        if person in have_trait:
            prob_trait[person] = True
        else:
            prob_trait[person] = False
    #Loop over all people in the prob_gene dictionary
    for person in prob_gene:
        probabilities[person]["gene"][prob_gene[person]]+=p
    #Loop over all people in the prob_trait dictionary
    for person in prob_trait:
        probabilities[person]["trait"][prob_trait[person]]+=p




def normalize(probabilities):
    """
    Update `probabilities` such that each probability distribution
    is normalized (i.e., sums to 1, with relative proportions the same).
    """
    #Loop over all the probabilities
    for person in probabilities:
        for dist in probabilities[person]:
            total = sum(probabilities[person][dist].values())
            for val in probabilities[person][dist]:
                probabilities[person][dist][val] /= total


if __name__ == "__main__":
    main()
