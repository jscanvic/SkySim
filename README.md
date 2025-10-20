# SkySim

Basic sky simulation built with physical accuracy in mind

![Simulated sky](sky.jpeg)

## How to use?

1. Run the command `python simulate.py` in your terminal.
2. Open the generated `sky.jpeg` in one of the supported viewers.

## Supported viewers

* https://photo-sphere-viewer.js.org/playground.html

## Sky models

**Perez All-Weather**

All-weather model for sky luminance distribution—Preliminary configuration and validation, Perez et al., Solar Energy (vol. 50, issue 3), 1993

**Preetham**

A practical analytic model for daylight, Preetham et al., Proceedings of the 26th annual conference on Computer graphics and interactive techniques (SIGGRAPH '99), 1999

## TODO

* [ ] Make a proper command-line interface
* [ ] Use a better scale for the final dynamic range
* [ ] Compute the position of the sun from time and location
* [ ] Make an interactive viewer
* [ ] Add more popular sky models
