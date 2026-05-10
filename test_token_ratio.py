import sys
import os
import time
import numpy as np
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.tools.funcs import cal_token

SUFFIX = "\nJust output ok without any other content"

CONFIGS = [
    {
        "name": "kimi_baseline",
        "llm_name": "kimi-k2.6",
        "key": "sk-XCIS2SQrnnyFYQZpB2VKBEWLKDR1y5t6WxmYyZGWQIH9T6Ws",
        "base_url": "https://api.moonshot.cn/v1",
    },
    {
        "name": "doubao",
        "llm_name": "ep-20250612104210-ss27q",
        "key": "02002c34-ca56-4797-a742-87fd6ec3981c",
        "base_url": "https://ark-cn-beijing.bytedance.net/api/v3",
    },
    {
        "name": "kimi2.5",
        "llm_name": "kimi-k2.5",
        "key": "sk-t7iwfuuGv42FbRhkOaSLjczH8VN9BpnhD62qHpcstzdDNS5r",
        "base_url": "https://api.moonshot.cn/v1",
    },
    {
        "name": "mimo-v2.5",
        "llm_name": "mimo-v2.5-pro",
        "key": "sk-c9bb0kdfpbjcq9pbovudukaw2xbh1wwd1izl17wj001dvtlq",
        "base_url": "https://api.xiaomimimo.com/v1",
    },
]

REAL_TEXTS = [
    "Artificial intelligence has transformed the way we interact with technology. From voice assistants to recommendation systems, AI is embedded in our daily lives.",

    "The history of computing dates back to the abacus, but modern computing began with the invention of the transistor in 1947. Since then, Moore's Law has driven exponential growth in processing power, enabling everything from personal computers to smartphones. Today, quantum computing promises to revolutionize the field once again by solving problems that classical computers cannot handle efficiently. Researchers at major tech companies and universities are racing to achieve quantum advantage, where quantum computers outperform classical ones on practical tasks.",

    "Climate change represents one of the most significant challenges facing humanity in the 21st century. Rising global temperatures are causing glaciers to melt, sea levels to rise, and weather patterns to become more extreme. The Intergovernmental Panel on Climate Change has warned that without significant reductions in greenhouse gas emissions, global temperatures could rise by more than 1.5 degrees Celsius above pre-industrial levels within the next two decades. This would lead to devastating consequences including more frequent droughts, floods, and heatwaves. Transitioning to renewable energy sources such as solar, wind, and hydroelectric power is essential. Additionally, carbon capture technologies and reforestation efforts can help mitigate existing atmospheric carbon dioxide. Governments worldwide must implement policies that incentivize sustainable practices while ensuring economic stability.",

    "Software engineering is a discipline that encompasses the systematic design, development, testing, and maintenance of software systems. It draws upon principles from computer science, mathematics, and project management to create reliable and efficient software solutions. The software development lifecycle typically includes several phases: requirements gathering, system design, implementation, testing, deployment, and maintenance. Agile methodologies have become increasingly popular, emphasizing iterative development, continuous feedback, and adaptive planning. Version control systems like Git enable teams to collaborate effectively on large codebases. Continuous integration and continuous deployment pipelines automate the build, test, and release processes, reducing human error and accelerating delivery. Code reviews ensure quality standards are maintained, while automated testing frameworks validate functionality across different scenarios. Microservices architecture has gained traction as a way to build scalable and maintainable systems, allowing teams to develop, deploy, and scale individual components independently.",

    "The human brain is arguably the most complex organ in the known universe. Weighing approximately 1.4 kilograms, it contains roughly 86 billion neurons, each forming thousands of synaptic connections with other neurons. This intricate network gives rise to consciousness, memory, emotion, and all higher cognitive functions. Neuroscience has made remarkable strides in understanding brain function through techniques such as functional magnetic resonance imaging, electroencephalography, and optogenetics. These tools allow researchers to observe brain activity in real time and even manipulate specific neural circuits. The field of neuroplasticity has revealed that the brain is not a static organ but continuously rewires itself in response to experience, learning, and injury. This discovery has profound implications for rehabilitation after stroke, treatment of mental health disorders, and educational practices. Meanwhile, brain-computer interfaces are being developed to help individuals with paralysis regain communication and motor control. Companies like Neuralink are working on implantable devices that could eventually allow direct communication between the brain and computers, raising both exciting possibilities and serious ethical questions about privacy, autonomy, and the nature of human identity.",

    "The global economy is undergoing a profound transformation driven by digital technologies, shifting geopolitical landscapes, and evolving consumer preferences. E-commerce has disrupted traditional retail, with platforms like Amazon and Alibaba reshaping how goods are bought and sold worldwide. The rise of remote work, accelerated by the COVID-19 pandemic, has created new economic patterns as workers relocate from expensive urban centers to more affordable regions. Cryptocurrency and blockchain technology have introduced decentralized financial systems that challenge traditional banking and monetary policy. Meanwhile, supply chain disruptions caused by the pandemic, natural disasters, and geopolitical tensions have highlighted the fragility of just-in-time manufacturing and globalized production networks. Many companies are now pursuing nearshoring and diversification strategies to build more resilient supply chains. Inflationary pressures in major economies have prompted central banks to raise interest rates, creating ripple effects across housing markets, stock valuations, and emerging market debt sustainability. The transition to a green economy is creating new industries and jobs in renewable energy, electric vehicles, and sustainable agriculture, while simultaneously threatening traditional fossil fuel sectors. Income inequality continues to widen in many countries, fueling political polarization and social unrest. Policymakers face the difficult challenge of balancing economic growth with environmental sustainability and social equity. International trade agreements are being renegotiated to address concerns about intellectual property protection, labor standards, and data privacy. The World Trade Organization struggles to maintain relevance as bilateral and regional trade agreements proliferate.",

    "The Renaissance was a transformative period in European history that spanned roughly from the 14th to the 17th century. Originating in Italy, particularly in city-states like Florence, Venice, and Milan, it marked a renewed interest in classical Greek and Roman art, literature, and philosophy. The Medici family in Florence played a crucial role as patrons of the arts, supporting luminaries such as Leonardo da Vinci, Michelangelo, and Botticelli. The invention of the printing press by Johannes Gutenberg around 1440 revolutionized the dissemination of knowledge, making books more affordable and accessible to a broader audience. This technological breakthrough facilitated the spread of humanist ideas and contributed to the Protestant Reformation initiated by Martin Luther in 1517. In science, figures like Nicolaus Copernicus, Galileo Galilei, and Johannes Kepler challenged the geocentric model of the universe, laying the groundwork for modern astronomy and physics. The period also saw significant advances in anatomy and medicine, with Andreas Vesalius publishing his groundbreaking work on human anatomy. In politics, Niccolo Machiavelli wrote The Prince, a seminal work on political strategy and power that remains influential today. The Renaissance also transformed architecture, with Filippo Brunelleschi designing the dome of the Florence Cathedral and Andrea Palladio developing a classical architectural style that would influence buildings across Europe and the Americas for centuries. In literature, Dante Alighieri, Petrarch, and Giovanni Boccaccio established the Tuscan dialect as the standard for Italian literature, while William Shakespeare and Miguel de Cervantes produced works that would become cornerstones of English and Spanish literature respectively. The era exploration saw Christopher Columbus, Vasco da Gama, and Ferdinand Magellan embark on voyages that expanded European understanding of the world and initiated the Age of Discovery. The Renaissance fundamentally altered the intellectual and cultural landscape of Europe, setting the stage for the Scientific Revolution and the Enlightenment that followed. Its legacy continues to influence art, science, politics, and culture around the world, reminding us of the enduring power of human creativity and intellectual curiosity.",
]


def test_llm(config, text):
    client = OpenAI(api_key=config["key"], base_url=config["base_url"])
    full_text = text + SUFFIX
    try:
        r = client.chat.completions.create(
            model=config["llm_name"],
            messages=[{"role": "user", "content": full_text}],
        )
        input_tokens = r.usage.prompt_tokens
        return input_tokens
    except Exception as e:
        print(f"  ERROR for {config['name']}: {e}")
        return None


def main():
    results = {c["name"]: [] for c in CONFIGS}

    for i, text in enumerate(REAL_TEXTS):
        openai_tokens = cal_token(text + SUFFIX)
        print(f"\n=== Text #{i+1}: {len(text)} chars, cal_token: {openai_tokens} ===")
        print(f"    Preview: {text[:80]}...")

        for config in CONFIGS:
            print(f"  Testing {config['name']} ({config['llm_name']})...")
            input_tokens = test_llm(config, text)
            if input_tokens is not None:
                results[config["name"]].append((openai_tokens, input_tokens))
                print(f"    cal_token={openai_tokens}, llm_input_tokens={input_tokens}")
            time.sleep(1)

    print("\n\n========== RESULTS ==========")
    for config in CONFIGS:
        name = config["name"]
        data = results[name]
        if len(data) < 2:
            print(f"{name}: Not enough data points")
            continue

        x_arr = np.array([d[0] for d in data], dtype=float)
        y_arr = np.array([d[1] for d in data], dtype=float)

        A = np.vstack([x_arr, np.ones(len(x_arr))]).T
        k, b = np.linalg.lstsq(A, y_arr, rcond=None)[0]

        residuals = y_arr - (k * x_arr + b)
        rmse = np.sqrt(np.mean(residuals ** 2))

        print(f"\n{name} ({config['llm_name']}):")
        print(f"  Data points: {data}")
        print(f"  k = {k:.6f}")
        print(f"  b = {b:.6f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  Formula: llm_tokens = {k:.6f} * cal_token + {b:.6f}")


if __name__ == "__main__":
    main()
