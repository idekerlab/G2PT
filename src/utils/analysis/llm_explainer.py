import openai
import pygraphviz as pgv


class EpistaticInteractionExplainer:
    def __init__(self, api_key, model='chatgpt-4o-latest'):
        """Initialize with the OpenAI API key."""
        self.api_key = api_key
        self.model = model

    @staticmethod
    def create_prompt(biological_system, phenotype, gene1, gene2):
        """Create the prompt based on the given biological system and gene information."""
        prompt = f"""
        You are an expert in genetics and systems biology. Given a pair of genes that function within the same biological system, analyze their potential epistatic interaction based on their Gene Ontology (GO) annotations.

        **Biological System (GO Annotation):** {biological_system}
        **Gene 1:** {gene1}
        **Gene 2:** {gene2})

        **Task:**
        1. Describe the biological roles of both genes within the specified system.
        2. Explain how their interaction influences the **{phenotype}** phenotype.
        3. Identify possible epistatic relationships (e.g., suppression, redundancy, synthetic lethality).
        4. Predict the phenotypic effects of single-gene and double-gene perturbations.
        """
        return prompt

    def explain_epistasis(self, biological_system, phenotype, gene1, gene2, temperature=0.7):
        """Make an API call to ChatGPT and explain the epistatic interaction for a given phenotype."""
        prompt = self.create_prompt(biological_system, phenotype, gene1, gene2)

        # Make the API call
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in genetics and systems biology."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            api_key=self.api_key
        )

        # Return the generated explanation
        return response["choices"][0]["message"]["content"]


    def generate_graphviz_code(self, epistasis_analysis):
        """Step 2: Ask ChatGPT to generate a Graphviz DOT diagram based on analysis."""
        prompt = f"""
        Based on the following epistasis analysis, generate a Graphviz DOT format diagram to visually represent the gene interactions.

        **Epistasis Analysis:**
        {epistasis_analysis}

        **Task:**
        - Use arrows and edge labels to indicate relationships (e.g., activation, suppression, redundancy).
        - Include an additional "Phenotype" node to illustrate the outcome.
        - Ensure the output is in **valid Graphviz DOT format** with no explanations or comments.

        **Example Output Format:**
        ```
        digraph Epistasis {{
            rankdir=TB;
            Gene1 [label="Gene1\n(Description)"];
            Gene2 [label="Gene2\n(Description)"];
            Phenotype [label="Phenotype", shape=box];

            Gene1 -> Gene2 [label="Activates"];
            Gene2 -> Phenotype [label="Suppresses"];
        }}
        ```
        """
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are an expert in genetics and systems biology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            api_key=self.api_key
        )

        return response["choices"][0]["message"]["content"]

    @staticmethod
    def generate_agraph(graphviz_code):
        """Step 3: Convert Graphviz DOT code to a PyGraphviz AGraph object."""
        graph = pgv.AGraph(string=graphviz_code)
        graph.layout(prog="dot")
        return graph

    def visualize_epistasis(self, biological_system, phenotype, gene1, gene2, output_file=None):
        """Run full pipeline: analyze epistasis, get Graphviz DOT, and render diagram."""
        print("Step 1: Analyzing epistasis...")
        epistasis_analysis = self.explain_epistasis(biological_system, phenotype, gene1, gene2)
        print("Epistasis Analysis Done!\n")

        print("Step 2: Generating Graphviz DOT Code...")
        graphviz_code = self.generate_graphviz_code(epistasis_analysis)
        print("Graphviz Code Generated!\n")

        print("Step 3: Rendering Epistasis Diagram...")
        epistasis_graph = self.generate_agraph(graphviz_code)
        if output_file is not None:
            epistasis_graph.draw(output_file)
            print(f"Diagram saved as {output_file}.")

        return epistasis_graph  # Returning PyGraphviz AGraph object

    def visualize_epistasis_from_llm_analysis(self, epistasis_analysis, output_file=None):
        graphviz_code = self.generate_graphviz_code(epistasis_analysis)
        graphviz_code = '\n'.join(graphviz_code.splitlines()[1:-1])
        epistasis_graph = self.generate_agraph(graphviz_code)
        if output_file is not None:
            epistasis_graph.draw(output_file)
            print(f"Diagram saved as {output_file}.")
        return epistasis_graph  # Returning PyGraphviz AGraph object