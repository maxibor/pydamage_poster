import streamlit as st
import numpy as np
import pydamage
import pandas as pd
import time
from fancy_cache import fancy_cache


@fancy_cache(unique_to_session=True)
def generate_ancient_data():

    res = {}

    p = np.random.uniform(0.01, 0.7)
    pmin = np.random.uniform(0.001, 0.02)
    pmax = np.random.uniform(0.06, 0.5)
    seqlen = 30
    x = np.arange(0, seqlen, 1)

    ancient_model = pydamage.models.damage_model()
    damage_amount = ancient_model.fit(x, p, pmin, pmax, wlen=seqlen)
    noise_sd = np.random.uniform(0.001, pmax / 5, 1)
    noise = np.random.normal(0, noise_sd, seqlen)
    damage_amount_noisy = damage_amount + noise
    damage_amount_noisy[damage_amount_noisy < 0] = 0

    res["x"] = x
    res["true damage"] = damage_amount
    res["oberserved damage"] = damage_amount_noisy
    res["seqlen"] = seqlen
    res["nb_c"] = np.random.randint(1000, 10000, seqlen)
    res["c_to_t"] = np.random.binomial(res["nb_c"], res["oberserved damage"])
    res["c_to_c"] = res["nb_c"] - res["c_to_t"]

    return res


def pydamage_guess():
    pass


def print_header():
    header = """
    <img src="https://raw.githubusercontent.com/maxibor/pydamage/master/docs/img/logo.png" alt="Pydamage logo" width="100%">
    """
    st.markdown(header, unsafe_allow_html=True)


def print_title():
    title = """# PyDamage: automated ancient damage identification and estimation for contigs in ancient DNA de novo assembly

### Maxime Borry<sup>1*</sup>, Alexander Hübner<sup>1,2</sup>, A.B. Rohrlach<sup>3,4</sup>, Christina Warinner<sup>1,2,5</sup>

<sup>1</sup><sub>Microbiome Sciences Group, Department of Archaeogenetics, Max Planck Institute for the Science of Human History, Jena, Germany, Kahlaische Straße 10, 07445 Jena, Germany</sub>  
<sup>2</sup><sub>Faculty of Biological Sciences, Friedrich-Schiller University, 07743, Jena, Germany</sub>  
<sup>3</sup><sub>Population Genetics Group, Department of Archaeogenetics, Max Planck Institute for the Science of Human History, Jena, Germany, Kahlaische Straße 10, 07445 Jena, Germany</sub>  
<sup>4</sup><sub>ARC Centre of Excellence for Mathematical and Statistical Frontiers, The University of Adelaide, Adelaide SA 5005, Australia</sub>  
<sup>5</sup><sub>Department of Anthropology, Harvard University, Cambridge, MA, USA 02138</sub>  
<sup>*</sup><sub>Corresponding author: [borry@shh.mpg.de](mailto:borry@shh.mpg.de)</sub>
    """
    st.markdown(title, unsafe_allow_html=True)


def print_intro():
    intro = """
## Introduction
DNA *de novo assembly* can be used to reconstruct longer stretches of DNA (contigs), including genes and even genomes, from short DNA sequencing reads. 
Applying this technique to metagenomic data derived from archaeological remains, such as paleofeces and dental calculus, we can investigate past microbiome functional diversity that may be absent or underrepresented in the modern microbiome gene catalogue. 
However, compared to modern samples, ancient samples are often burdened with environmental contamination, resulting in metagenomic datasets that represent mixtures of ancient and modern DNA. 
The ability to rapidly and reliably establish the authenticity and integrity of ancient samples is essential for ancient DNA studies, and the ability to distinguish between ancient and modern sequences is particularly important for ancient microbiome studies. 
Characteristic patterns of ancient DNA damage, namely DNA fragmentation and cytosine deamination (observed as C-to-T transitions) are typically used to authenticate ancient samples and sequences. 
However, existing tools for inspecting and filtering aDNA damage either compute it at the read level, which leads to high data loss and lower quality when used in combination with de novo assembly, or require manual inspection, which is impractical for ancient assemblies that typically contain tens to hundreds of thousands of contigs. 
To address these challenges, we designed [PyDamage](https://github.com/maxibor/pydamage), a robust, automated approach for aDNA damage estimation and authentication of de novo assembled aDNA. PyDamage uses a likelihood ratio based approach to discriminate between truly ancient contigs and contigs originating from modern contamination. 
We test PyDamage on both simulated, and empirical aDNA data from archaeological paleofeces, and we demonstrate its ability to reliably and automatically identify contigs bearing DNA damage characteristic of aDNA. 
Coupled with aDNA de novo assembly, PyDamage opens up new doors to explore functional diversity in ancient metagenomic datasets.
    """
    st.markdown(intro, unsafe_allow_html=True)


def get_user_input():
    user_input = st.slider(
        "Damage proportion on 5' end",
        min_value=0.0,
        max_value=0.99,
        value=0.5001234,
        step=0.01,
        help="Guess the amount of the damage of the 5' end (x=0)",
    )
    return float(user_input)


# @st.cache(suppress_st_warning=True)
def display_result(guess, true_value):
    if round(guess, 2) == round(true_value, 2):
        st.write("Nice")
    else:
        st.write("Try again !")


def game_intro():
    intro = """## Can you Beat PyDamage 🎮  🎯 🧬 ☠️ ?
PyDamage is a program to make a statistical estimation of the amount of damage carried by
DNA *de novo* assembled contigs.  
But before you read further, do you think you can beat PyDamage
at guessing the amount of damage shown below ?
    """
    st.markdown(intro, unsafe_allow_html=True)
    run_game()
    restart_game()


def run_game():
    d = generate_ancient_data()
    df = pd.DataFrame(d)
    st.line_chart(df.loc[:, ["oberserved damage", "true damage"]])
    guess = get_user_input()
    st.write(guess)
    st.write(round(d["true damage"][0], 2))
    if st.button("Guess"):
        display_result(guess, d["true damage"][0])


def restart_game():
    if st.button("Restart game"):
        st.caching.clear_cache()
        raise st.experimental_rerun()


if __name__ == "__main__":
    # st.caching.clear_cache()
    print_header()
    print_title()
    game_intro()
    print_intro()
