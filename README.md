# Deep Learning with R

## Overview

This repository contains various notebooks for learning about deep learning with `torch` and `mlr3torch` in R.
A total of 7 notebooks are available as quarto files in the `notebooks` folder.
For each notebook, there is also a corresponding quarto notebook with exercises, which also contains the solutions (to be unfolded).

## Setup

In order to be able to successfully run the notebooks and solve the exercises, make sure that you have the right environment set up.

### R Version

These notebooks were developed using R 4.4.2, so ideally you should use the same version.
You can check your current R version by running `R --version` in your terminal.
If you have the same R version, you can skip the rest of this section.

A convenient way to simultaneously maintain different versions of R and to easily switch between them is to use `rig`, the R installation manager.
It's [documentation](https://github.com/r-lib/rig?tab=readme-ov-file#id-installation) contains instructions on how to install it on Windows, macOS, and Linux.

After having installed `rig`, you can install R 4.4.2 and make it the default by running the following command in your terminal:

```{bash, eval = FALSE}
rig add 4.4.2
rig default 4.4.2
```

Verify that the R version is 4.4.2 by running `R --version` in your terminal.

### Libraries

For managing libraries, we use the [`renv` package](https://rstudio.github.io/renv/articles/renv.html).
If you don't have `renv` installed, you can install it by running the following command:

```{r, eval = FALSE}
install.packages("renv")
```

Next, initialize the `renv` environment by running the following command:

```{r, eval = FALSE}
renv::init()
```

Finally, restore the renv environment by running the following command.
This might take some time as it downloads all the required libraries.

```{r, eval = FALSE}
renv::restore()
```

To be able to use the `torch` package, you need to also run this additional command:

```{r}
torch::install_torch()
```

If this completes successfully, you are ready to go! :rocket:

## Rendering the notebooks

TODO: Make-command
