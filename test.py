from rag import RAG


def test_basic_functionality():
    """Test basic RAG functionality with a simple document"""
    # Initialize RAG
    rag = RAG()

    # Test document
    document = """
The Evolution of Artificial Intelligence and Its Impact on Society

Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, revolutionizing various aspects of human life and society. From its humble beginnings in the 1950s to today's sophisticated systems, AI has undergone remarkable evolution and continues to shape our future in unprecedented ways.

The Historical Development of AI

The concept of artificial intelligence first gained significant attention during the Dartmouth Conference of 1956, where pioneering researchers gathered to discuss the possibility of creating machines that could "think." Early AI research focused on general problem solving and symbolic methods. In the 1960s, researchers explored natural language processing and developed programs that could understand basic English sentences.

The 1970s saw the emergence of expert systems, which were designed to solve complex problems by reasoning through bodies of knowledge, primarily expressed as if-then rules. XCON, developed by Digital Equipment Corporation, became one of the first successful commercial expert systems, saving the company millions of dollars in operations costs.

The AI winter of the 1970s and early 1980s marked a period of reduced funding and interest in AI research due to unmet expectations and limitations of the existing computing power. However, the field experienced a resurgence in the 1990s with the introduction of machine learning approaches based on neural networks and probability theory.

Modern AI Technologies and Applications

Today's AI systems leverage deep learning and neural networks to achieve unprecedented capabilities in various domains. Computer vision systems can now recognize faces and objects with remarkable accuracy, while natural language processing models can generate human-like text and engage in sophisticated conversations.

In healthcare, AI algorithms assist doctors in diagnosing diseases, analyzing medical images, and predicting patient outcomes. The technology has shown particular promise in radiology, where AI systems can detect subtle patterns in X-rays and MRI scans that might escape human notice. Researchers are also developing AI-powered drug discovery platforms that can significantly accelerate the process of identifying new therapeutic compounds.

The financial sector has embraced AI for fraud detection, algorithmic trading, and risk assessment. Machine learning models analyze vast amounts of transaction data in real-time to identify suspicious patterns and prevent fraudulent activities. AI-driven robo-advisors are democratizing investment management by providing automated, low-cost portfolio management services to individual investors.

Impact on Employment and the Workforce

The integration of AI into various industries has sparked intense debate about its impact on employment. While AI automation has displaced some jobs, particularly in manufacturing and data processing, it has also created new opportunities in AI development, data science, and robot maintenance.

Studies suggest that rather than wholesale job replacement, AI is more likely to augment human capabilities and transform existing roles. Workers will need to develop new skills to work alongside AI systems effectively. This has led to increased emphasis on continuous learning and adaptation in the workforce.

Ethical Considerations and Challenges

As AI systems become more sophisticated and autonomous, important ethical questions have emerged. Issues of privacy, bias, and accountability are at the forefront of discussions about AI deployment. For instance, facial recognition systems have faced scrutiny for potential biases across different demographic groups, while autonomous vehicles raise complex questions about decision-making in life-or-death situations.

Data privacy concerns have become particularly acute as AI systems require vast amounts of data for training. Organizations must balance the benefits of AI with the need to protect individual privacy and comply with regulations like GDPR and CCPA.

The challenge of explaining AI decisions, known as the "black box" problem, remains a significant concern, especially in critical applications like healthcare and criminal justice. Researchers are actively working on developing more interpretable AI models and better methods for explaining their decisions.

Future Prospects and Societal Implications

Looking ahead, artificial general intelligence (AGI) remains a long-term goal of AI research. Unlike current AI systems that excel at specific tasks, AGI would possess human-like general problem-solving abilities. While estimates vary widely, most experts believe AGI is still decades away from realization.

Quantum computing could dramatically accelerate AI capabilities by solving complex optimization problems more efficiently than classical computers. This could lead to breakthroughs in areas like drug discovery, climate modeling, and financial risk assessment.

The integration of AI into education systems is expected to revolutionize learning through personalized instruction and adaptive assessment. AI tutors could provide customized feedback and adjust teaching methods based on individual student needs and learning styles.

Environmental Applications and Sustainability

AI is playing an increasingly important role in addressing environmental challenges. Machine learning models help optimize energy grids, predict weather patterns, and improve resource management. AI-powered systems are being used to monitor deforestation, track wildlife populations, and develop more efficient renewable energy solutions.

Smart city initiatives leverage AI to reduce energy consumption, improve traffic flow, and enhance waste management. These applications demonstrate how AI can contribute to sustainable urban development and environmental conservation efforts.

Conclusions and Future Outlook

The continued evolution of AI technology promises to bring both opportunities and challenges for society. Success in navigating this transformation will require thoughtful consideration of ethical implications, proactive policy development, and inclusive dialogue among stakeholders.

As AI capabilities continue to advance, maintaining human agency and ensuring equitable access to AI benefits will be crucial. The technology's potential to address global challenges like climate change, healthcare accessibility, and educational inequality suggests that its impact will only grow in importance in the coming decades.

The key to successful AI integration lies in striking the right balance between technological advancement and human values, ensuring that AI development serves the collective good while respecting individual rights and promoting social justice.
    """

    print("\n1. Testing document addition and chunking...")
    rag.add_documents([document])
    print(f"Number of chunks created: {len(rag.chunks)}")
    print("\nFirst chunk preview:")
    print(rag.chunks[0]["text"][:100], "...\n")

    test_queries = [
        "What is the history of artificial intelligence?",
        "How does AI impact employment?",
        "What are the ethical challenges of AI?",
        "How is AI used in environmental applications?",
        "What is the future outlook for AI?",
    ]

    print("2. Testing search functionality with multiple queries...")
    for query in test_queries:
        print(f"\nSearch results for: {query}")
        results = rag.search(query, top_k=2)
        for i, (text, score) in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"Score: {score:.4f}")
            print(f"Text: {text[:200]}...")


def test_multiple_documents():
    """Test RAG with multiple documents"""
    rag = RAG()

    sections = [
        """The Evolution of Artificial Intelligence and Its Impact on Society
Artificial Intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, revolutionizing various aspects of human life and society. From its humble beginnings in the 1950s to today's sophisticated systems, AI has undergone remarkable evolution and continues to shape our future in unprecedented ways.""",
        """
The Historical Development of AI
The concept of artificial intelligence first gained significant attention during the Dartmouth Conference of 1956, where pioneering researchers gathered to discuss the possibility of creating machines that could "think." Early AI research focused on general problem solving and symbolic methods. In the 1960s, researchers explored natural language processing and developed programs that could understand basic English sentences.
The 1970s saw the emergence of expert systems, which were designed to solve complex problems by reasoning through bodies of knowledge, primarily expressed as if-then rules. XCON, developed by Digital Equipment Corporation, became one of the first successful commercial expert systems, saving the company millions of dollars in operations costs.
The AI winter of the 1970s and early 1980s marked a period of reduced funding and interest in AI research due to unmet expectations and limitations of the existing computing power. However, the field experienced a resurgence in the 1990s with the introduction of machine learning approaches based on neural networks and probability theory.
""",
        """Modern AI Technologies and Applications
Today's AI systems leverage deep learning and neural networks to achieve unprecedented capabilities in various domains. Computer vision systems can now recognize faces and objects with remarkable accuracy, while natural language processing models can generate human-like text and engage in sophisticated conversations.
In healthcare, AI algorithms assist doctors in diagnosing diseases, analyzing medical images, and predicting patient outcomes. The technology has shown particular promise in radiology, where AI systems can detect subtle patterns in X-rays and MRI scans that might escape human notice. Researchers are also developing AI-powered drug discovery platforms that can significantly accelerate the process of identifying new therapeutic compounds.
The financial sector has embraced AI for fraud detection, algorithmic trading, and risk assessment. Machine learning models analyze vast amounts of transaction data in real-time to identify suspicious patterns and prevent fraudulent activities. AI-driven robo-advisors are democratizing investment management by providing automated, low-cost portfolio management services to individual investors.""",
        """
Impact on Employment and the Workforce
The integration of AI into various industries has sparked intense debate about its impact on employment. While AI automation has displaced some jobs, particularly in manufacturing and data processing, it has also created new opportunities in AI development, data science, and robot maintenance.
Studies suggest that rather than wholesale job replacement, AI is more likely to augment human capabilities and transform existing roles. Workers will need to develop new skills to work alongside AI systems effectively. This has led to increased emphasis on continuous learning and adaptation in the workforce.""",
        """
Ethical Considerations and Challenges
As AI systems become more sophisticated and autonomous, important ethical questions have emerged. Issues of privacy, bias, and accountability are at the forefront of discussions about AI deployment. For instance, facial recognition systems have faced scrutiny for potential biases across different demographic groups, while autonomous vehicles raise complex questions about decision-making in life-or-death situations.
Data privacy concerns have become particularly acute as AI systems require vast amounts of data for training. Organizations must balance the benefits of AI with the need to protect individual privacy and comply with regulations like GDPR and CCPA.
The challenge of explaining AI decisions, known as the "black box" problem, remains a significant concern, especially in critical applications like healthcare and criminal justice. Researchers are actively working on developing more interpretable AI models and better methods for explaining their decisions.""",
        """
Future Prospects and Societal Implications
Looking ahead, artificial general intelligence (AGI) remains a long-term goal of AI research. Unlike current AI systems that excel at specific tasks, AGI would possess human-like general problem-solving abilities. While estimates vary widely, most experts believe AGI is still decades away from realization.
Quantum computing could dramatically accelerate AI capabilities by solving complex optimization problems more efficiently than classical computers. This could lead to breakthroughs in areas like drug discovery, climate modeling, and financial risk assessment.
The integration of AI into education systems is expected to revolutionize learning through personalized instruction and adaptive assessment. AI tutors could provide customized feedback and adjust teaching methods based on individual student needs and learning styles.""",
        """
Environmental Applications and Sustainability
AI is playing an increasingly important role in addressing environmental challenges. Machine learning models help optimize energy grids, predict weather patterns, and improve resource management. AI-powered systems are being used to monitor deforestation, track wildlife populations, and develop more efficient renewable energy solutions.
Smart city initiatives leverage AI to reduce energy consumption, improve traffic flow, and enhance waste management. These applications demonstrate how AI can contribute to sustainable urban development and environmental conservation efforts.""",
        """
Conclusions and Future Outlook
The continued evolution of AI technology promises to bring both opportunities and challenges for society. Success in navigating this transformation will require thoughtful consideration of ethical implications, proactive policy development, and inclusive dialogue among stakeholders.
As AI capabilities continue to advance, maintaining human agency and ensuring equitable access to AI benefits will be crucial. The technology's potential to address global challenges like climate change, healthcare accessibility, and educational inequality suggests that its impact will only grow in importance in the coming decades.
The key to successful AI integration lies in striking the right balance between technological advancement and human values, ensuring that AI development serves the collective good while respecting individual rights and promoting social justice.""",
    ]

    print("\n3. Testing multiple section handling...")
    rag.add_documents(sections)
    print(f"Number of chunks created: {len(rag.chunks)}")

    print("\n4. Testing cross-section search...")
    query = "How does AI impact jobs and what are the ethical concerns?"
    results = rag.search(query, top_k=3)
    print("\nSearch results for:", query)
    for i, (text, score) in enumerate(results, 1):
        print(f"\nResult {i}:")
        print(f"Score: {score:.4f}")
        print(f"Text: {text[:200]}...")


if __name__ == "__main__":
    print("Starting RAG Pipeline Tests...")

    try:
        print("\n=== Basic Functionality Test ===")
        test_basic_functionality()

        print("\n=== Multiple Documents Test ===")
        test_multiple_documents()

    except Exception as e:
        print(f"\nError occurred: {str(e)}")

    print("\nTests completed.")
