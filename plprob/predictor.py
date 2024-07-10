"""Predictor for computing CP probability using historical data."""

import numpy as np
import pandas as pd

from datetime import datetime

from dateutil.relativedelta import relativedelta
from typing import List, Dict, Set, Iterable, Optional

from .model import get_asset_list, PlModel, PlError


class PlPredictor:
    """
    A class for computing CP probability using asset actuals and forecasts.

    Attributes
    ----------
        asset_list : List[str]
            The assets for which scenarios will be generated.
        num_of_assets : int
            How many assets there are.

        hist_actual_df, hist_forecast_df : pd.DataFrame
            Historical actual and forecasted values for the assets.
        scen_start_time : pd.TimeStamp
            When the generated scenarios will start.
        
        model : Optional[GeminiModel]
            The model fitted on historical data that will be used to generate
            scenarios.

        forecast_resolution_in_minute : int
            The frequency of the intervals at which forecasts are provided.
        num_of_horizons : int
            How many forecast time intervals to generate scenarios for.
        forecast_lead_hours : int
            The time gap between when forecasts are issued and the beginning of
            the period which they are predicting.

        forecast_issue_time : pd.Timestamp
            When forecasts were issued for the period for which scenarios will
            be generated. E.g. if scenarios are to be generated for 2020-06-03,
            this might be 2020-06-02 12:00:00.
        scen_end_time : pd.Timestamp
            The end of the period for which scenarios will be generated.
        scen_timesteps : List[pd.Timestamp]
            The time points which generated scenarios will provide values for.

        forecasts : Optiona[Dict[pd.Series]]
            The forecasted values for the scenario time window which were used
            as a basis to generate scenarios.
        scenarios : Optional[Dict[pd.DataFrame]]
            The scenarios generated using this engine.
    """

    def __init__(self,
                 hist_actual_df: pd.DataFrame, hist_forecast_df: pd.DataFrame,
                 scen_start_time: pd.Timestamp,
                 num_of_cps: int,
                 hist_cps: List[int], 
                 forecast_resolution_in_minute: int = 60,
                 num_of_horizons: int = 24,
                 forecast_lead_time_in_hour: int = 12) -> None:

        # check that the dataframes with actual and forecast values are in the
        # right format, get the names of the assets they contain values for
        self.asset_list = get_asset_list(hist_actual_df, hist_forecast_df)
        self.num_of_assets = len(self.asset_list)

        self.hist_actual_df = hist_actual_df
        self.hist_forecast_df = hist_forecast_df
        self.scen_start_time = scen_start_time
        self.hist_start = self.hist_forecast_df.Forecast_time.min()
        self.hist_end = self.hist_forecast_df.Forecast_time.max()

        self.num_of_cps = num_of_cps
        self.hist_cps = [cp for cp in hist_cps]

        self.model = None

        self.forecast_resolution_in_minute = forecast_resolution_in_minute
        self.num_of_horizons = num_of_horizons
        self.forecast_lead_hours = forecast_lead_time_in_hour

        # figure out when forecasts for the time period for which scenarios
        # will be generated were issued
        self.forecast_issue_time = (
            self.scen_start_time - pd.Timedelta(self.forecast_lead_hours,
                                                unit='h')
            )

        # calculate the close of the window for which scenarios will be made
        self.scen_end_time = (
                self.scen_start_time
                + pd.Timedelta((self.num_of_horizons - 1)
                               * self.forecast_resolution_in_minute,
                               unit='min')
            )

        # get the time points at which scenario values will be generated
        self.scen_timesteps = pd.date_range(
            start=self.scen_start_time, end=self.scen_end_time,
            freq=str(self.forecast_resolution_in_minute) + 'min'
            ).tolist()

        self.solar_zone_mean = None
        self.solar_zone_std = None

        self.forecasts = None
        self.scenarios = None

        self.cp_prob = dict()
        self.peak_hour_prob = dict()

    def fit(self,
            asset_rho: float, horizon_rho: float,
            nearest_days: Optional[int] = None) -> None:
        """
        This function creates and fits a scenario model using historical asset
        values. The model will estimate the distributions of the deviations
        from actual values observed in the forecast dataset.

        Arguments
        ---------
            asset_rho
                Hyper-parameter governing how strongly non-zero interactions
                between generators are penalized.
            horizon_rho
                Hyper-parameter governing how strongly non-zero interactions
                between time points are penalized.

            nearest_days
                If given, will not use historical asset values more than this
                number of days away from the given date in each year.
        """

        if nearest_days:
            dev_index = self.get_yearly_date_range(use_date=self.scen_start_time,
                                              num_of_days=nearest_days)
        else:
            dev_index = None

        ## use gpd for load
        use_gpd = True

        self.model = PlModel(self.scen_start_time, self.get_hist_df_dict(),
                                 None, dev_index,
                                 self.forecast_resolution_in_minute,
                                 self.num_of_horizons,
                                 self.forecast_lead_hours, use_gpd=use_gpd)

        self.model.fit(asset_rho, horizon_rho)

    def create_scenario(self,
                        nscen: int, forecast_df: pd.DataFrame,
                        **gpd_args) -> None:
        """
        This function generates a number of scenarios using the given forecasts
        and the model that has been fit on historical asset values.

        Arguments
        ---------
            nscen
                How many scenarios to generate.
            forecast_df
                Forecasted asset values that will be added to the deviations
                generated by the model to produce scenarios.
            gpd_args
                Optional arguments to pass to `fit_conditional_marginal_dist`.

        """
        if self.model is None:
            raise PlError(
                "Cannot generate scenarios until a model has been fitted!")

        self.model.get_forecast(forecast_df)
        
        upper_dict = None

        self.model.generate_gauss_scenarios(nscen, upper_dict=upper_dict)
        self.scenarios = self.model.scen_df
        self.forecasts = self.get_forecast(forecast_df)

    def get_hist_df_dict(
            self,
            assets: Optional[Iterable[str]] = None
            ) -> Dict[str, pd.DataFrame]:
        """Utility for getting historical values for a given set of assets."""

        if assets is None:
            assets = self.asset_list
        else:
            assets = sorted(assets)

        return {
            'actual': self.hist_actual_df.loc[:, assets],
            'forecast': self.hist_forecast_df.loc[
                        :, ['Issue_time', 'Forecast_time'] + assets]
            }

    def get_forecast(self, forecast_df: pd.DataFrame) -> pd.Series:
        """Get forecasts issued for the period scenarios are generated for."""

        use_forecasts = forecast_df[
            forecast_df['Issue_time'] == self.forecast_issue_time].drop(
                columns='Issue_time').set_index('Forecast_time')

        use_forecasts = use_forecasts[
            (use_forecasts.index >= self.scen_start_time)
            & (use_forecasts.index <= self.scen_end_time)
            ].sort_index()
        use_forecasts.index = self.scen_timesteps

        return use_forecasts.unstack()

    def get_yearly_date_range(
            self,
            use_date: pd.Timestamp, num_of_days: Optional[int] = None,
            ) -> Set[pd.Timestamp]:
        """Gets a historical date range around a given day and time for model training.

        Arguments
        ---------
            use_date        The date and time around which the range will be centered.
            num_of_days     The "radius" of the range. If not given, all
                            historical days will be used instead.

        """
        hist_dates = set(pd.date_range(
            start=self.hist_start, end=self.hist_end, freq='D', tz='utc'))

        if num_of_days is not None:

            near_dates = set()
            relative_years = {hist_date.year - use_date.year for hist_date in hist_dates}

            for relative_year in relative_years:
                year_date = use_date + relativedelta(years = relative_year)

                near_dates.update(pd.date_range(
                    start=year_date - pd.Timedelta(num_of_days, unit='D'),
                    periods=2 * num_of_days + 1, freq='D', tz='utc')
                    )

            hist_dates &= near_dates

        return sorted(list(hist_dates))
    

    def update_cp(self, cp):
        self.hist_cps.append(cp)
        self.hist_cps = sorted(self.hist_cps)[-self.num_of_cps:]

    def compute_cp_probs(self):
        if self.scenarios is None:
            raise PlError(
                "Generate scenarios before computing CP probability!")
        
        # Compute probability of being new CP day
        cp_bins = self.hist_cps + [np.inf]
        if cp_bins[0] != 0:
            cp_bins = [0] + cp_bins
        
        cp_count = pd.cut(self.scenarios.max(axis=1), 
                           cp_bins, 
                           labels=list(range(len(cp_bins) - 2, -1, -1)))
        self.cp_prob = dict(cp_count.groupby(cp_count, observed=False).count()[::-1].cumsum() / len(cp_count))
        
        # Compute probabilities of peak hour
        peak_hour = self.scenarios.droplevel(axis=1,level=0).idxmax(axis=1)
        self.peak_hour_prob = dict(peak_hour.groupby(peak_hour).count() / len(peak_hour))