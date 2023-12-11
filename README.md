# Design-and-development-of-a-focused-web-crawler

– In this project, we designed and developed a focused web crawler with the integration of the BERT model to perform data extraction and find out cosine similarity scores. The main objective of this project is to retrieve and analyze relevant content from 50 website links based on a specific keyword and compare their content with the seed page to define the similarity between them. The developed web crawler implements BERT – a state-of-the-art transformer-based language model for its advanced natural language processing abilities. BERT enables the system to understand the context and nuances of the search term, ensuring the extraction of accurate and contextually relevant data from all targeted website links. The designed web crawler efficiently navigates through web pages, extracts textual content, and identifies links related to the specified keyword. On top of this, the system employs embeddings and cosine similarity to compute similarity scores between the extracted content from each of the 50 websites and the seed page. Embeddings, generated by the BERT model, capture semantic representations of text, allowing for robust comparisons between the web page content. Cosine similarity serves as a measure to quantify the similarity between the embeddings, revealing the extent to which each website’s content aligns with that of the seed page. The experimental results show the effectiveness and efficiency of the proposed approach in extracting relevant data from multiple web sources and evaluating their similarity with the seed page. The system's accuracy and performance are evaluated using precision, recall, and F1- score metrics, indicating its capability to extract pertinent information. The computed similarity scores provide insights into the relatedness of web pages with respect to the seed page, aiding in the content comparison and relevance assessment.
