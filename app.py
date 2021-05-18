import streamlit as st
import numpy as np
import pydamage
import pandas as pd
import time
from fancy_cache import fancy_cache
import time


@fancy_cache(unique_to_session=True, ttl=600, suppress_st_warning=True)
def generate_ancient_data():

    res = {}

    p = np.random.uniform(0.01, 0.7)
    pmin = np.random.uniform(0.001, 0.02)
    pmax = np.random.uniform(0.0011, 0.5)
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
    res["observed damage"] = damage_amount_noisy
    res["seqlen"] = seqlen
    res["nb_c"] = np.random.randint(1000, 10000, seqlen)
    res["c_to_t"] = np.random.binomial(res["nb_c"], res["observed damage"])
    res["c_to_c"] = res["nb_c"] - res["c_to_t"]

    return res


@fancy_cache(unique_to_session=True, ttl=600, suppress_st_warning=True)
def pydamage_guess(damage, mut_count, conserved_count):
    from pydamage.model_fit import fit_models

    model_A = pydamage.models.damage_model()
    model_B = pydamage.models.null_model()

    res = fit_models(
        ref="test",
        model_A=model_A,
        model_B=model_B,
        damage=damage,
        mut_count=mut_count,
        conserved_count=conserved_count,
        verbose=False,
    )

    for i in range(30):
        res.update({f"CtoT-{i}": damage[i]})
    res.update({"reference": "PyDamage example"})
    res.update({"coverage": 4})
    res.update({"c_to_t": mut_count.sum()})
    res.update({"nb_c": mut_count.sum() + conserved_count.sum()})

    return res


def print_header():
    header = """
    <img src="https://raw.githubusercontent.com/maxibor/pydamage/master/docs/img/logo.png" alt="Pydamage logo" width="100%">
    """
    st.markdown(header, unsafe_allow_html=True)


def print_title():
    title = """# PyDamage: automated ancient damage identification and estimation for contigs in ancient DNA de novo assembly

### [Maxime Borry](https://twitter.com/notmaxib)<sup>1*</sup>, [Alexander Hübner](https://twitter.com/alexhbnr)<sup>1,2</sup>, [A.B. Rohrlach](https://twitter.com/BRohrlach)<sup>3,4</sup>, [Christina Warinner](https://twitter.com/twarinner?lang=en)<sup>1,2,5</sup>

<sup>1</sup><sub>Microbiome Sciences Group, Department of Archaeogenetics, Max Planck Institute for the Science of Human History, Jena, Germany, Kahlaische Straße 10, 07445 Jena, Germany</sub>  
<sup>2</sup><sub>Faculty of Biological Sciences, Friedrich-Schiller University, 07743, Jena, Germany</sub>  
<sup>3</sup><sub>Population Genetics Group, Department of Archaeogenetics, Max Planck Institute for the Science of Human History, Jena, Germany, Kahlaische Straße 10, 07445 Jena, Germany</sub>  
<sup>4</sup><sub>ARC Centre of Excellence for Mathematical and Statistical Frontiers, The University of Adelaide, Adelaide SA 5005, Australia</sub>  
<sup>5</sup><sub>Department of Anthropology, Harvard University, Cambridge, MA, USA 02138</sub>  
<sup>*</sup><sub>Corresponding author: [borry@shh.mpg.de](mailto:borry@shh.mpg.de)</sub>
    """
    st.markdown(title, unsafe_allow_html=True)


def flash_talk():
    jitsi = """
**Chat with Maxime on [vchat.eva.mpg.de/pydamage_poster](https://vchat.eva.mpg.de/pydamage_poster) - 02/06/2021 - 12:30-15:30 CET **  
    """
    st.markdown(jitsi)
    url = "https://youtu.be/jYxsKQznAuY"
    with st.beta_expander("Watch the Flash Talk"):
        st.video(url)


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


@fancy_cache(unique_to_session=True, ttl=600, suppress_st_warning=True)
def display_result(guess, true_value, pydamage_result, duration):

    pydamage_diff = abs(round(pydamage_result["pmax"], 2) - round(true_value, 2))
    guess_diff = abs(round(guess, 2) - round(true_value, 2))
    both_diff = round(abs(guess_diff - pydamage_diff) * 100)

    st.markdown(
        f"""
- True value of the Damage on 5' end: **{round(true_value,2)}** 
- PyDamage guessed (in **{duration}** ms): **{round(pydamage_result["pmax"],2)}**
- PyDamage computed **pvalue** for the whole sequence (*H0*: No Damage): **{round(pydamage_result["pvalue"],2)}**
- Your guess: **{round(guess, 2)}**
    """
    )
    if pydamage_diff == guess_diff:
        st.markdown(
            f"""
**Congratulations, you were as good as PyDamage**, but were you as fast 😉 ?    
To congratulate you, we may still have [beautiful PyDamage stickers](https://raw.githubusercontent.com/maxibor/pydamage/master/docs/img/logo.png) available, just send me a DM on twitter [(@notmaxib)](https://twitter.com/notmaxib) !
And please read further to understand how PyDamage works !
        """
        )
    elif guess_diff < pydamage_diff:
        st.markdown(
            f"""
**Congratulations, you were better than PyDamage by {both_diff}%**, that's why there are still humans behing the wheel, right ? 😉  
To congratulate you, we may still have [beautiful PyDamage stickers](https://raw.githubusercontent.com/maxibor/pydamage/master/docs/img/logo.png) available, just send me a DM on twitter [(@notmaxib)](https://twitter.com/notmaxib) !  
And please read further to understand how PyDamage works !
        """
        )
    elif guess_diff > pydamage_diff:
        st.markdown(
            """
Sorry, PyDamage beat you this time, but you can always try again by refreshing the page 😉  
Read further to understand how PyDamage works !
        """
        )
    with st.beta_expander("See details"):
        st.markdown(
            f"""
For the purpose of this poster, simulated data were simulated using Numpy, with parameters picked randomly.  
The code for this simulation is available [here](https://github.com/maxibor/pydamage_poster/blob/68289d18f3df784987419392f4f89a0c3d014232/app.py#L11)
#### Details of data used for PyDamage computation:
- Number of "C" sites: {pydamage_result["nb_c"]}
- Out of these, number of "C to T" substitution events used for damage estimation: {pydamage_result["c_to_t"]}

*The standalone PyDamage estimation is usually faster, but there is a delay with demonstration poster because of the Streamlit backend.*
        """
        )
        # st.pyplot(pydamage_plot)


def game_intro():
    intro = """# Can you Beat PyDamage 🎮  🎯 🧬 ☠️ ?
[PyDamage](https://github.com/maxibor/pydamage) is a program to make a statistical estimation of the amount of damage carried by
DNA *de novo* assembled contigs.  
But before you read further, do you think you can beat PyDamage
at guessing the amount of damage shown below ?
    """
    st.markdown(intro, unsafe_allow_html=True)
    run_game()
    # restart_game()


def run_game():
    d = generate_ancient_data()
    t0 = time.time()
    pydam = pydamage_guess(
        damage=d["observed damage"],
        mut_count=d["c_to_t"],
        conserved_count=d["c_to_c"],
    )
    t1 = time.time()
    duration = round((t1 - t0) * 1000, 2)
    df = pd.DataFrame(d)
    st.line_chart(df.loc[:, "observed damage"])
    guess = get_user_input()
    # st.write(guess)
    # st.write(round(d["true damage"][0], 2))
    if st.button("Guess"):
        display_result(guess, d["true damage"][0], pydam, duration)


def restart_game():
    if st.button("Restart game"):
        st.caching.clear_cache()
        raise st.experimental_rerun()


def introduction():
    intro = """
# Introduction
- DNA *de novo assembly* can be used to reconstruct longer stretches of DNA (contigs), including genes and even genomes, from short DNA sequencing reads. 
- However, compared to modern samples, ancient samples are often burdened with environmental contamination, resulting in mixtures of ancient and modern DNA. 
- Distinguishing between ancient and modern sequences is particularly important for ancient microbiome studies. 
- Characteristic patterns of ancient DNA damage, namely DNA fragmentation and cytosine deamination (observed as C-to-T transitions) are typically used to authenticate ancient samples and sequences. 
- Existing tools for inspecting and filtering aDNA damage do not scale the thousands of reference generated by *de novo* assembly. 

We introduce [**PyDamage**](https://github.com/maxibor/pydamage), an automated approach for aDNA damage estimation and authentication of de novo assembled aDNA.  
It uses a likelihood ratio based approach to discriminate between truly ancient contigs and contigs originating from modern contamination.   
We tested PyDamage on both simulated, and empirical aDNA data from archaeological paleofeces.  
We demonstrated its ability to reliably and automatically identify contigs bearing DNA damage characteristic of aDNA.   
    """
    st.markdown(intro, unsafe_allow_html=True)


def methods():
    methods = """
# Methods
- PyDamage takes an alignment file as input (`SAM`,`BAM`,`CRAM`)
- Counting of C to T transitions is done thanks to [PySam](https://github.com/pysam-developers/pysam)
- At each position of the reads, the damage is modelled with a binomial distribution distribution $B(n,p)$ (figure 1)  
    - the parameter $n$ is the number of *C* sites at each position
    - the parameter $p$ is estimated at each position by fitting two models to observed CtoT transitions data, with least square optimization.  
        - a null model (no damage), that assumes a uniform rate of rate of C to T substitutions (figure 1, green line)  
        - a damage model, that assumes a decreasing rate of C to T substitutions (figure 2, red line)  

<img src="https://raw.githubusercontent.com/maxibor/pydamage_poster/master/img/ridgeplot.png" alt="Pydamage logo" width="100%">

**Figure 1**: Example distribution of the C to T transitions at each site, modelled as a binomial distribution. The red line represents the fitted damage model, while the green line represents the fitted null model.  



- The log likelihood is computed under each model, and a likelihood ratio test is performed.
- A pvalue is computed from the likelihood ratio test. 

    """
    st.markdown(methods, unsafe_allow_html=True)


def results():
    results = """
# Results
## Simulated data

We tested PyDamage on simulated data, varying the coverage, GC content, amount of damage, reference length, and read length (figure 2).  

<img src="https://raw.githubusercontent.com/maxibor/pydamage_poster/master/img/simulation.png" alt="Simulation scheme" width="100%">  

**Figure 2**: Simulation scheme for evaluating the performance of PyDamage. **a** The GC content of the three microbial reference genomes.
**b** The read length distributions used as input into gargammel *fragSim*. 
**c** The amount of aDNA damage as observed as the frequency of C-to-T substitutions on the terminal 5' end of the DNA fragments that was added using gargammel *deamSim*.
**d** Nine coverage bins from which the exact coverage was sampled by randomly drawing a number from the uniform distribution defining the bin. **e** Nine contig length bins from which the exact contig length was sampled by randomly drawing a number from the uniform distribution defining the bin.


After different Generalized Linear Model (GLM), 
and selecting the best performing model using balanced accuracy and Nagelkerke’s $R^2$, 
we found that **only coverage, amount of damage, and reference length** greatly affected the performance of PyDamage.  

Thanks to this GML model, we were also able to provide the user with an estimation of how accurate a PyDamage prediction can be (figure 3).

<img src="https://raw.githubusercontent.com/maxibor/pydamage_poster/master/img/glm.png" alt="GLM accuracy model" width="100%">  

**Figure 3**: PyDamage predicted model accuracy of simulated data, in function of coverage, reference length, and amount of damage. Light blue indicates improved model accuracy,
 with parameter combinations resulting in better than 50% accuracy are outlined in green.

## Archeological data

Finally, we applied PyDamage on the ZSM028 samples (previously published in [Borry et al. 2020](https://peerj.com/articles/9001/)) 
assembled with [metaSPAdes](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5411777/).  
The assembly yielded a total of  359,807 contigs, with an N50 of 429 bp. 
Such assemblies, consisting of a large number of relatively short contigs, are typical for *de novo* assembled aDNA datasets ([Wibowo et al., 2021](https://www.nature.com/articles/s41586-021-03532-0))
Out of the 17,103 contigs longer than 1000bp, PyDamage estimated the damage for 99.75% of them (17,061 contigs).  

After filtering resulting contigs with enough coverage for significant pvalues, 
1944 contigs were statistically identified by PyDamage as carrying C to T deamination damage, with a mean 5' damage of 14.3% (figure 4).

<img src="https://raw.githubusercontent.com/maxibor/pydamage_poster/master/img/real_data.png" alt="damage real data" width="100%">  

**Figure 4**: Damage profile of PyDamage filtered contigs of ZSM028. The center line is the mean, the shaded area is $\pm$ one standard-deviation around the mean.
    """
    st.markdown(results, unsafe_allow_html=True)


def conclusion():
    conclusion = """
# Conclusion

We demonstrated that PyDamage has highly reliable damage prediction accuracy for sequences with high coverage, long lengths, and high damage, 
but the tool’s power to assess damage is reduced for lower coverage, shorter contigs length, and lower deamination levels.   
While this might at first seem like an issue, in practice, in *de novo* assembly, no contig below ~6X coverage can be assembled, and shorter contigs  are usually discarded by the end user.  

aDNA damage levels (cytosine deamination and fragmentation) are features of the DNA itself and out of the researcher’s control, but researchers can improve assembly and model accuracy through deeper sequencing.  
As the fields of microbiology and evolutionary biology increasingly turn to the archaeological records to investigate the rich and dynamic evolutionary history of ancient microbial communities, 
it has become vital to develop tools for assembling and authenticating ancient metagenomic DNA.   
**Coupled with aDNA de novoassembly, PyDamage opens up new doors to explore and understand the functional diversity of ancient metagenomes.**


The PyDamage preprint is available on BioRxiv: [10.1101/2021.03.24.436838v1](https://www.biorxiv.org/content/10.1101/2021.03.24.436838v1)  

The PyDamage software is avaible on GitHub: [github.com/maxibor/pydamage](https://github.com/maxibor/pydamage)
    """
    st.markdown(conclusion, unsafe_allow_html=True)


def references():
    references = """
# References
<sup>Borry, M., Cordova, B., Perri, A., Wibowo, M., Honap, T. P., Ko, J., Yu, J., Britton, K., Girdland-Flink, L., Power, R. C., Stuijts, I., Salazar-Garc ́ıa, D. C., Hofman, C., Hagan, R., Kagon ́e, T. S., Meda, N., Carabin, H., Jacobson, D., Reinhard, K., Lewis, C., Kostic, A., Jeong, C., Herbig, A., Huebner, A.,and Warinner, C. (2020). Coproid predicts the source of coprolites and paleofeces using microbiome composition and host dna content.PeerJ, 8:e9001. Publisher: PeerJ Inc.</sup>  

<sup>Nurk, S., Meleshko, D., Korobeynikov, A., and Pevzner, P. A. (2017).  metaspades:  a new versatile metagenomic assembler.Genome research, 27(5):824–834.</sup>  

<sup>Wibowo, M. C., Yang, Z., Borry, M., H ̈ubner, A., Huang, K. D., Tierney, B. T., Zimmerman, S., Barajas-Olmos, F., Contreras-Cubas, C., Garcia-Ortiz, H., Martinez-Hernandez, A., Luber, J. M., Kirstahler, P.,Blohm, T., Smiley, F. E., Arnold, R., Ballall, S. A., Pamp, S. J., Russ, J., Maixner, F., Rota-Stabelli, O.,Segata, N., Reinhard, K., Orozco, L., Warinner, C., Snow, M., LeBlanc, S., and Kostic, A. D. (2021). Reconstruction of ancient microbial genomes from the human gut. Nature</sup>   

    """
    st.markdown(references, unsafe_allow_html=True)


if __name__ == "__main__":
    # st.caching.clear_cache()
    print_header()
    print_title()
    flash_talk()
    game_intro()
    introduction()
    methods()
    results()
    conclusion()
    references()
