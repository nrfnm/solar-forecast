# ☀️solar-forecast

For a forecasting challenge at [energy-arena.org](https://energy-arena.org)

Developed as part of a seminar at Karlsruher Institut für Technologie ([KIT](https://kit.edu)), Institut für Industriebetriebslehre und Industrielle Produktion ([IIP](https://www.iip.kit.edu))
### Clustering
Sourcing data about installed Solar Capacity via the _open_mastr_ Package from the Markstammdatenregister. Using 
PLZ Coordinate Map Sourced from [WZBSocialScienceCenter](https://github.com/WZBSocialScienceCenter/plz_geocoord) to fill in mostly missing GPS data.
Provided a script to generate K-Means Cluster.

### Integration
To integrate the model with _energy-arena_ [starer repo](https://github.com/zubasa107/energy-arena-participate) from energy-arena.org clone both repos in the same directory side by side.
Setup the starter repo as described in its documentation (API...),
copy custom_model.py from this repo to the stater repo.

To automate on your VM (or other devices) set up as described in the _starter repo_ (e.g. cronjob)

If desired, you can archive the actual NWP Forecasts to do proper EMOS Calibration. Use the script ```run_collect_ensemble.sh```