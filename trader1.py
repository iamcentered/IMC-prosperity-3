from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
import string
import jsonpickle
import numpy as np
import math


class Product:
    AMETHYSTS = "RAINFOREST_RESIN"
    STARFRUIT = "KELP"
    SQUIDINK = "SQUID_INK"
    GIFT_BASKET_1 = "PICNIC_BASKET1"
    GIFT_BASKET_2 = "PICNIC_BASKET2"
    CROISSANTS = "CROISSANTS"
    JAMS = "JAMS"
    DJEMBE = "DJEMBES"
    SYNTHETIC = "SYNTHETIC"
    SPREAD1 = "SPREAD1"
    SPREAD2 = "SPREAD2"
    COCONUT = "VOLCANIC_ROCK"
    COCONUT_COUPON = "VOLCANIC_ROCK_VOUCHER_9500"
    MAGNIFICENT_MACARONS = "MAGNIFICENT_MACARONS"


PARAMS = {
    Product.AMETHYSTS: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0.5,
        "volume_limit": 0,
    },
    Product.STARFRUIT: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "starfruit_min_edge": 2,
    },
    Product.SQUIDINK: {
        "take_width": 2,  # Wider width for trend following
        "clear_width": 1,
        "prevent_adverse": False,  # Allow larger orders
        "adverse_volume": 20,
        "trend_beta": 0.5,  # Momentum factor
        "disregard_edge": 2,
        "join_edge": 1,
        "default_edge": 3,
    },
    Product.SPREAD1: {
        "default_spread_mean": 17000.343325282296,
        "default_spread_std": 56.53114441467583,
        "spread_std_window": 30,
        "zscore_threshold": 1.2,
        "target_position": 60,
    },
    Product.SPREAD2: {
        "default_spread_mean": 10000.406921817692,
        "default_spread_std": 91.48029024709751,
        "spread_std_window": 30,
        "zscore_threshold": 3,
        "target_position": 60,
    },
    Product.COCONUT_COUPON: {
        "mean_volatility": 0.020472021403370906,
        "threshold": 0.0007174092199465122,
        "strike": 10500,
        "starting_time_to_expiry": 247 / 250,
        "std_window": 6,
        "zscore_threshold": 21,
    },
    Product.MAGNIFICENT_MACARONS: {
        "arb_edge": 2,
        "convert_limit": 10,
        "feature_coefs": {
            "sugarPrice": 0.1,
            "sunlightIndex": 0.05,
            "importTariff": -0.2,
            "exportTariff": 0.2,
            "transportFees": -0.1
        },
        "feature_offset": 0.0
    }
}

BASKET_WEIGHTS_1 = {
    Product.CROISSANTS: 6,
    Product.JAMS: 3,
    Product.DJEMBE: 1,
}
BASKET_WEIGHTS_2 = {
    Product.CROISSANTS: 4,
    Product.JAMS: 2,
}

from math import log, sqrt, exp
from statistics import NormalDist


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.AMETHYSTS: 50,
            Product.STARFRUIT: 50,
            Product.SQUIDINK: 50,
            Product.GIFT_BASKET_1: 60,
            Product.GIFT_BASKET_2: 100,
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBE: 60,
            Product.COCONUT: 400,
            Product.COCONUT_COUPON: 200,
            Product.MAGNIFICENT_MACARONS: 75
        }
    # def take_best_orders(
    #     self,
    #     product: str,
    #     fair_value: int,
    #     take_width: float,
    #     orders: List[Order],
    #     order_depth: OrderDepth,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    #     prevent_adverse: bool = False,
    #     adverse_volume: int = 0,
    # ) -> (int, int):
    #     position_limit = self.LIMIT[product]
    #     if len(order_depth.sell_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -1 * order_depth.sell_orders[best_ask]

    #         if best_ask <= fair_value - take_width:
    #             quantity = min(
    #                 best_ask_amount, position_limit - position
    #             )  # max amt to buy
    #             if quantity > 0:
    #                 orders.append(Order(product, best_ask, quantity))
    #                 buy_order_volume += quantity
    #                 order_depth.sell_orders[best_ask] += quantity
    #                 if order_depth.sell_orders[best_ask] == 0:
    #                     del order_depth.sell_orders[best_ask]

    #     if len(order_depth.buy_orders) != 0:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
    #         if best_bid >= fair_value + take_width:
    #             quantity = min(
    #                 best_bid_amount, position_limit + position
    #             )  # should be the max we can sell
    #             if quantity > 0:
    #                 orders.append(Order(product, best_bid, -1 * quantity))
    #                 sell_order_volume += quantity
    #                 order_depth.buy_orders[best_bid] -= quantity
    #                 if order_depth.buy_orders[best_bid] == 0:
    #                     del order_depth.buy_orders[best_bid]
    #     return buy_order_volume, sell_order_volume

    # def take_best_orders_with_adverse(
    #     self,
    #     product: str,
    #     fair_value: int,
    #     take_width: float,
    #     orders: List[Order],
    #     order_depth: OrderDepth,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    #     adverse_volume: int,
    # ) -> (int, int):

    #     position_limit = self.LIMIT[product]
    #     if len(order_depth.sell_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_ask_amount = -1 * order_depth.sell_orders[best_ask]
    #         if abs(best_ask_amount) <= adverse_volume:
    #             if best_ask <= fair_value - take_width:
    #                 quantity = min(
    #                     best_ask_amount, position_limit - position
    #                 )  # max amt to buy
    #                 if quantity > 0:
    #                     orders.append(Order(product, best_ask, quantity))
    #                     buy_order_volume += quantity
    #                     order_depth.sell_orders[best_ask] += quantity
    #                     if order_depth.sell_orders[best_ask] == 0:
    #                         del order_depth.sell_orders[best_ask]

    #     if len(order_depth.buy_orders) != 0:
    #         best_bid = max(order_depth.buy_orders.keys())
    #         best_bid_amount = order_depth.buy_orders[best_bid]
    #         if abs(best_bid_amount) <= adverse_volume:
    #             if best_bid >= fair_value + take_width:
    #                 quantity = min(
    #                     best_bid_amount, position_limit + position
    #                 )  # should be the max we can sell
    #                 if quantity > 0:
    #                     orders.append(Order(product, best_bid, -1 * quantity))
    #                     sell_order_volume += quantity
    #                     order_depth.buy_orders[best_bid] -= quantity
    #                     if order_depth.buy_orders[best_bid] == 0:
    #                         del order_depth.buy_orders[best_bid]

    #     return buy_order_volume, sell_order_volume
    
    # def market_make(
    #     self,
    #     product: str,
    #     orders: List[Order],
    #     bid: int,
    #     ask: int,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    # ) -> (int, int):
    #     buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
    #     if buy_quantity > 0:
    #         orders.append(Order(product, round(bid), buy_quantity))  # Buy order

    #     sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
    #     if sell_quantity > 0:
    #         orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
    #     return buy_order_volume, sell_order_volume

    # def clear_position_order(
    #     self,
    #     product: str,
    #     fair_value: float,
    #     width: int,
    #     orders: List[Order],
    #     order_depth: OrderDepth,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    # ) -> List[Order]:
    #     position_after_take = position + buy_order_volume - sell_order_volume
    #     fair_for_bid = round(fair_value - width)
    #     fair_for_ask = round(fair_value + width)

    #     buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
    #     sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

    #     if position_after_take > 0:
    #         # Aggregate volume from all buy orders with price greater than fair_for_ask
    #         clear_quantity = sum(
    #             volume
    #             for price, volume in order_depth.buy_orders.items()
    #             if price >= fair_for_ask
    #         )
    #         clear_quantity = min(clear_quantity, position_after_take)
    #         sent_quantity = min(sell_quantity, clear_quantity)
    #         if sent_quantity > 0:
    #             orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
    #             sell_order_volume += abs(sent_quantity)

    #     if position_after_take < 0:
    #         # Aggregate volume from all sell orders with price lower than fair_for_bid
    #         clear_quantity = sum(
    #             abs(volume)
    #             for price, volume in order_depth.sell_orders.items()
    #             if price <= fair_for_bid
    #         )
    #         clear_quantity = min(clear_quantity, abs(position_after_take))
    #         sent_quantity = min(buy_quantity, clear_quantity)
    #         if sent_quantity > 0:
    #             orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
    #             buy_order_volume += abs(sent_quantity)

    #     return buy_order_volume, sell_order_volume

    # def starfruit_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
    #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_bid = max(order_depth.buy_orders.keys())
    #         filtered_ask = [
    #             price
    #             for price in order_depth.sell_orders.keys()
    #             if abs(order_depth.sell_orders[price])
    #             >= self.params[Product.STARFRUIT]["adverse_volume"]
    #         ]
    #         filtered_bid = [
    #             price
    #             for price in order_depth.buy_orders.keys()
    #             if abs(order_depth.buy_orders[price])
    #             >= self.params[Product.STARFRUIT]["adverse_volume"]
    #         ]
    #         mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
    #         mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
    #         if mm_ask == None or mm_bid == None:
    #             if traderObject.get("starfruit_last_price", None) == None:
    #                 mmmid_price = (best_ask + best_bid) / 2
    #             else:
    #                 mmmid_price = traderObject["starfruit_last_price"]
    #         else:
    #             mmmid_price = (mm_ask + mm_bid) / 2

    #         if traderObject.get("starfruit_last_price", None) != None:
    #             last_price = traderObject["starfruit_last_price"]
    #             last_returns = (mmmid_price - last_price) / last_price
    #             pred_returns = (
    #                 last_returns * self.params[Product.STARFRUIT]["reversion_beta"]
    #             )
    #             fair = mmmid_price + (mmmid_price * pred_returns)
    #         else:
    #             fair = mmmid_price
    #         traderObject["starfruit_last_price"] = mmmid_price
    #         return fair
    #     return None

    # def make_amethyst_orders(
    #     self,
    #     order_depth: OrderDepth,
    #     fair_value: int,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    #     volume_limit: int,
    # ) -> (List[Order], int, int):
    #     orders: List[Order] = []
    #     baaf = min(
    #         [
    #             price
    #             for price in order_depth.sell_orders.keys()
    #             if price > fair_value + 1
    #         ]
    #     )
    #     bbbf = max(
    #         [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
    #     )

    #     if baaf <= fair_value + 2:
    #         if position <= volume_limit:
    #             baaf = fair_value + 3  # still want edge 2 if position is not a concern

    #     if bbbf >= fair_value - 2:
    #         if position >= -volume_limit:
    #             bbbf = fair_value - 3  # still want edge 2 if position is not a concern

    #     buy_order_volume, sell_order_volume = self.market_make(
    #         Product.AMETHYSTS,
    #         orders,
    #         bbbf + 1,
    #         baaf - 1,
    #         position,
    #         buy_order_volume,
    #         sell_order_volume,
    #     )
    #     return orders, buy_order_volume, sell_order_volume
    # # def squid_ink_ma_crossover(self, order_depth: OrderDepth, traderObject) -> float:
    # #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    # #         best_ask = min(order_depth.sell_orders.keys())
    # #         best_bid = max(order_depth.buy_orders.keys())
    # #         mid_price = (best_ask + best_bid) / 2

    # #         # Maintain price history for 1000 ticks
    # #         price_history = traderObject.get("squid_ink_price_history", [])
    # #         price_history.append(mid_price)

    # #         # Keep only last 1000 prices
    # #         if len(price_history) > 1000:
    # #             price_history.pop(0)
    # #         traderObject["squid_ink_price_history"] = price_history

    # #         # Initialize default signal
    # #         traderObject["squid_ink_signal"] = traderObject.get("squid_ink_signal", "HOLD")

    # #         # Calculate indicators
    # #         if len(price_history) >= 1000:
    # #             ma_1000 = sum(price_history) / 1000
    # #             prev_ma = traderObject.get("squid_ink_prev_ma", ma_1000)
    # #             prev_price = traderObject.get("squid_ink_prev_price", mid_price)

    # #             # Detect crossover events
    # #             if prev_price < prev_ma and mid_price > ma_1000:
    # #                 traderObject["squid_ink_signal"] = "BUY"
    # #             elif prev_price > prev_ma and mid_price < ma_1000:
    # #                 traderObject["squid_ink_signal"] = "SELL"

    # #             # Update previous values
    # #             traderObject["squid_ink_prev_ma"] = ma_1000
    # #             traderObject["squid_ink_prev_price"] = mid_price

    # #         return mid_price
    # #     return None
    # def squidink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
    #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_bid = max(order_depth.buy_orders.keys())
    #         mid_price = (best_ask + best_bid) / 2

    #         # Maintain price history for 1000 ticks
    #         price_history = traderObject.get("squidink_price_history", [])
    #         price_history.append(mid_price)

    #         # Keep only last 1000 prices
    #         if len(price_history) > 1000:
    #             price_history.pop(0)
    #         traderObject["squidink_price_history"] = price_history

    #         # Initialize default signal
    #         traderObject["squidink_signal"] = traderObject.get("squidink_signal", "HOLD")

    #         # Calculate indicators
    #         if len(price_history) >= 1000:
    #             ma_1000 = sum(price_history) / 1000
    #             prev_ma = traderObject.get("squidink_prev_ma", ma_1000)
    #             prev_price = traderObject.get("squidink_prev_price", mid_price)

    #             # Detect crossover events
    #             if prev_price < prev_ma and mid_price > ma_1000:
    #                 traderObject["squidink_signal"] = "BUY"
    #             elif prev_price > prev_ma and mid_price < ma_1000:
    #                 traderObject["squidink_signal"] = "SELL"

    #             # Update previous values
    #             traderObject["squidink_prev_ma"] = ma_1000
    #             traderObject["squidink_prev_price"] = mid_price

    #         return ma_1000  # Return MA as fair value
    #     return None

    # def take_orders(
    #     self,
    #     product: str,
    #     order_depth: OrderDepth,
    #     fair_value: float,
    #     take_width: float,
    #     position: int,
    #     prevent_adverse: bool = False,
    #     adverse_volume: int = 0,
    # ) -> (List[Order], int, int):
    #     orders: List[Order] = []
    #     buy_order_volume = 0
    #     sell_order_volume = 0

    #     if prevent_adverse:
    #         buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
    #             product,
    #             fair_value,
    #             take_width,
    #             orders,
    #             order_depth,
    #             position,
    #             buy_order_volume,
    #             sell_order_volume,
    #             adverse_volume,
    #         )
    #     else:
    #         buy_order_volume, sell_order_volume = self.take_best_orders(
    #             product,
    #             fair_value,
    #             take_width,
    #             orders,
    #             order_depth,
    #             position,
    #             buy_order_volume,
    #             sell_order_volume,
    #         )
    #     return orders, buy_order_volume, sell_order_volume

    # def clear_orders(
    #     self,
    #     product: str,
    #     order_depth: OrderDepth,
    #     fair_value: float,
    #     clear_width: int,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    # ) -> (List[Order], int, int):
    #     orders: List[Order] = []
    #     buy_order_volume, sell_order_volume = self.clear_position_order(
    #         product,
    #         fair_value,
    #         clear_width,
    #         orders,
    #         order_depth,
    #         position,
    #         buy_order_volume,
    #         sell_order_volume,
    #     )
    #     return orders, buy_order_volume, sell_order_volume

    # def make_starfruit_orders(
    #     self,
    #     order_depth: OrderDepth,
    #     fair_value: float,
    #     min_edge: float,
    #     position: int,
    #     buy_order_volume: int,
    #     sell_order_volume: int,
    # ) -> (List[Order], int, int):
    #     orders: List[Order] = []
    #     aaf = [
    #         price
    #         for price in order_depth.sell_orders.keys()
    #         if price >= round(fair_value + min_edge)
    #     ]
    #     bbf = [
    #         price
    #         for price in order_depth.buy_orders.keys()
    #         if price <= round(fair_value - min_edge)
    #     ]
    #     baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
    #     bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
    #     buy_order_volume, sell_order_volume = self.market_make(
    #         Product.STARFRUIT,
    #         orders,
    #         bbbf + 1,
    #         baaf - 1,
    #         position,
    #         buy_order_volume,
    #         sell_order_volume,
    #     )

    #     return orders, buy_order_volume, sell_order_volume
    
    # def get_swmid(self, order_depth) -> float:
    #     best_bid = max(order_depth.buy_orders.keys())
    #     best_ask = min(order_depth.sell_orders.keys())
    #     best_bid_vol = abs(order_depth.buy_orders[best_bid])
    #     best_ask_vol = abs(order_depth.sell_orders[best_ask])
    #     return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
    #         best_bid_vol + best_ask_vol
    #     )
    
    # def get_synthetic_basket1_order_depth(
    #     self, order_depths: Dict[str, OrderDepth]
    # ) -> OrderDepth:
    #     # Constants
    #     CHOCOLATE_PER_BASKET = BASKET_WEIGHTS_1[Product.CROISSANTS]
    #     STRAWBERRIES_PER_BASKET = BASKET_WEIGHTS_1[Product.JAMS]
    #     ROSES_PER_BASKET = BASKET_WEIGHTS_1[Product.DJEMBE]

    #     # Initialize the synthetic basket order depth
    #     synthetic_order_price = OrderDepth()

    #     # Calculate the best bid and ask for each component
    #     chocolate_best_bid = (
    #         max(order_depths[Product.CROISSANTS].buy_orders.keys())
    #         if order_depths[Product.CROISSANTS].buy_orders
    #         else 0
    #     )
    #     chocolate_best_ask = (
    #         min(order_depths[Product.CROISSANTS].sell_orders.keys())
    #         if order_depths[Product.CROISSANTS].sell_orders
    #         else float("inf")
    #     )
    #     strawberries_best_bid = (
    #         max(order_depths[Product.JAMS].buy_orders.keys())
    #         if order_depths[Product.JAMS].buy_orders
    #         else 0
    #     )
    #     strawberries_best_ask = (
    #         min(order_depths[Product.JAMS].sell_orders.keys())
    #         if order_depths[Product.JAMS].sell_orders
    #         else float("inf")
    #     )
    #     roses_best_bid = (
    #         max(order_depths[Product.DJEMBE].buy_orders.keys())
    #         if order_depths[Product.DJEMBE].buy_orders
    #         else 0
    #     )
    #     roses_best_ask = (
    #         min(order_depths[Product.DJEMBE].sell_orders.keys())
    #         if order_depths[Product.DJEMBE].sell_orders
    #         else float("inf")
    #     )

    #     # Calculate the implied bid and ask for the synthetic basket
    #     implied_bid = (
    #         chocolate_best_bid * CHOCOLATE_PER_BASKET
    #         + strawberries_best_bid * STRAWBERRIES_PER_BASKET
    #         + roses_best_bid * ROSES_PER_BASKET
    #     )
    #     implied_ask = (
    #         chocolate_best_ask * CHOCOLATE_PER_BASKET
    #         + strawberries_best_ask * STRAWBERRIES_PER_BASKET
    #         + roses_best_ask * ROSES_PER_BASKET
    #     )

    #     # Calculate the maximum number of synthetic baskets available at the implied bid and ask
    #     if implied_bid > 0:
    #         chocolate_bid_volume = (
    #             order_depths[Product.CROISSANTS].buy_orders[chocolate_best_bid]
    #             // CHOCOLATE_PER_BASKET
    #         )
    #         strawberries_bid_volume = (
    #             order_depths[Product.JAMS].buy_orders[strawberries_best_bid]
    #             // STRAWBERRIES_PER_BASKET
    #         )
    #         roses_bid_volume = (
    #             order_depths[Product.DJEMBE].buy_orders[roses_best_bid]
    #             // ROSES_PER_BASKET
    #         )
    #         implied_bid_volume = min(
    #             chocolate_bid_volume, strawberries_bid_volume, roses_bid_volume
    #         )
    #         synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

    #     if implied_ask < float("inf"):
    #         chocolate_ask_volume = (
    #             -order_depths[Product.CROISSANTS].sell_orders[chocolate_best_ask]
    #             // CHOCOLATE_PER_BASKET
    #         )
    #         strawberries_ask_volume = (
    #             -order_depths[Product.JAMS].sell_orders[strawberries_best_ask]
    #             // STRAWBERRIES_PER_BASKET
    #         )
    #         roses_ask_volume = (
    #             -order_depths[Product.DJEMBE].sell_orders[roses_best_ask]
    #             // ROSES_PER_BASKET
    #         )
    #         implied_ask_volume = min(
    #             chocolate_ask_volume, strawberries_ask_volume, roses_ask_volume
    #         )
    #         synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

    #     return synthetic_order_price
    
    # def get_synthetic_basket2_order_depth(
    #     self, order_depths: Dict[str, OrderDepth]
    # ) -> OrderDepth:
    #     # Constants
    #     CHOCOLATE_PER_BASKET2 = BASKET_WEIGHTS_2[Product.CROISSANTS]
    #     STRAWBERRIES_PER_BASKET2 = BASKET_WEIGHTS_2[Product.JAMS]

    #     # Initialize the synthetic basket order depth
    #     synthetic_order_price = OrderDepth()

    #     # Calculate the best bid and ask for each component
    #     chocolate_best_bid = (
    #         max(order_depths[Product.CROISSANTS].buy_orders.keys())
    #         if order_depths[Product.CROISSANTS].buy_orders
    #         else 0
    #     )
    #     chocolate_best_ask = (
    #         min(order_depths[Product.CROISSANTS].sell_orders.keys())
    #         if order_depths[Product.CROISSANTS].sell_orders
    #         else float("inf")
    #     )
    #     strawberries_best_bid = (
    #         max(order_depths[Product.JAMS].buy_orders.keys())
    #         if order_depths[Product.JAMS].buy_orders
    #         else 0
    #     )
    #     strawberries_best_ask = (
    #         min(order_depths[Product.JAMS].sell_orders.keys())
    #         if order_depths[Product.JAMS].sell_orders
    #         else float("inf")
    #     )
    #     # Calculate the implied bid and ask for the synthetic basket
    #     implied_bid = (
    #         chocolate_best_bid * CHOCOLATE_PER_BASKET2
    #         + strawberries_best_bid * STRAWBERRIES_PER_BASKET2
    #     )
    #     implied_ask = (
    #         chocolate_best_ask * CHOCOLATE_PER_BASKET2
    #         + strawberries_best_ask * STRAWBERRIES_PER_BASKET2
    #     )

    #     # Calculate the maximum number of synthetic baskets available at the implied bid and ask
    #     if implied_bid > 0:
    #         chocolate_bid_volume = (
    #             order_depths[Product.CROISSANTS].buy_orders[chocolate_best_bid]
    #             // CHOCOLATE_PER_BASKET2
    #         )
    #         strawberries_bid_volume = (
    #             order_depths[Product.JAMS].buy_orders[strawberries_best_bid]
    #             // STRAWBERRIES_PER_BASKET2
    #         )
    #         implied_bid_volume = min(
    #             chocolate_bid_volume, strawberries_bid_volume
    #         )
    #         synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

    #     if implied_ask < float("inf"):
    #         chocolate_ask_volume = (
    #             -order_depths[Product.CROISSANTS].sell_orders[chocolate_best_ask]
    #             // CHOCOLATE_PER_BASKET2
    #         )
    #         strawberries_ask_volume = (
    #             -order_depths[Product.JAMS].sell_orders[strawberries_best_ask]
    #             // STRAWBERRIES_PER_BASKET2
    #         )
    #         implied_ask_volume = min(
    #             chocolate_ask_volume, strawberries_ask_volume
    #         )
    #         synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

    #     return synthetic_order_price

    # def convert_synthetic_basket1_orders(
    #     self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    # ) -> Dict[str, List[Order]]:
    #     # Initialize the dictionary to store component orders
    #     component_orders = {
    #         Product.CROISSANTS: [],
    #         Product.JAMS: [],
    #         Product.DJEMBE: [],
    #     }

    #     # Get the best bid and ask for the synthetic basket
    #     synthetic_basket_order_depth = self.get_synthetic_basket1_order_depth(
    #         order_depths
    #     )
    #     best_bid = (
    #         max(synthetic_basket_order_depth.buy_orders.keys())
    #         if synthetic_basket_order_depth.buy_orders
    #         else 0
    #     )
    #     best_ask = (
    #         min(synthetic_basket_order_depth.sell_orders.keys())
    #         if synthetic_basket_order_depth.sell_orders
    #         else float("inf")
    #     )

    #     # Iterate through each synthetic basket order
    #     for order in synthetic_orders:
    #         # Extract the price and quantity from the synthetic basket order
    #         price = order.price
    #         quantity = order.quantity

    #         # Check if the synthetic basket order aligns with the best bid or ask
    #         if quantity > 0 and price >= best_ask:
    #             # Buy order - trade components at their best ask prices
    #             chocolate_price = min(
    #                 order_depths[Product.CROISSANTS].sell_orders.keys()
    #             )
    #             strawberries_price = min(
    #                 order_depths[Product.JAMS].sell_orders.keys()
    #             )
    #             roses_price = min(order_depths[Product.DJEMBE].sell_orders.keys())
    #         elif quantity < 0 and price <= best_bid:
    #             # Sell order - trade components at their best bid prices
    #             chocolate_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
    #             strawberries_price = max(
    #                 order_depths[Product.JAMS].buy_orders.keys()
    #             )
    #             roses_price = max(order_depths[Product.DJEMBE].buy_orders.keys())
    #         else:
    #             # The synthetic basket order does not align with the best bid or ask
    #             continue

    #         # Create orders for each component
    #         chocolate_order = Order(
    #             Product.CROISSANTS,
    #             chocolate_price,
    #             quantity * (BASKET_WEIGHTS_1[Product.CROISSANTS]),
    #         )
    #         strawberries_order = Order(
    #             Product.JAMS,
    #             strawberries_price,
    #             quantity * BASKET_WEIGHTS_1[Product.JAMS],
    #         )
    #         roses_order = Order(
    #             Product.DJEMBE, roses_price, quantity * BASKET_WEIGHTS_1[Product.DJEMBE]
    #         )

    #         # Add the component orders to the respective lists
    #         component_orders[Product.CROISSANTS].append(chocolate_order)
    #         component_orders[Product.JAMS].append(strawberries_order)
    #         component_orders[Product.DJEMBE].append(roses_order)

    #     return component_orders
    
    # def convert_synthetic_basket2_orders(
    #     self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    # ) -> Dict[str, List[Order]]:
    #     # Initialize the dictionary to store component orders
    #     component_orders = {
    #         Product.CROISSANTS: [],
    #         Product.JAMS: [],
    #     }

    #     # Get the best bid and ask for the synthetic basket
    #     synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(
    #         order_depths
    #     )
    #     best_bid = (
    #         max(synthetic_basket_order_depth.buy_orders.keys())
    #         if synthetic_basket_order_depth.buy_orders
    #         else 0
    #     )
    #     best_ask = (
    #         min(synthetic_basket_order_depth.sell_orders.keys())
    #         if synthetic_basket_order_depth.sell_orders
    #         else float("inf")
    #     )

    #     # Iterate through each synthetic basket order
    #     for order in synthetic_orders:
    #         # Extract the price and quantity from the synthetic basket order
    #         price = order.price
    #         quantity = order.quantity

    #         # Check if the synthetic basket order aligns with the best bid or ask
    #         if quantity > 0 and price >= best_ask:
    #             # Buy order - trade components at their best ask prices
    #             chocolate_price = min(
    #                 order_depths[Product.CROISSANTS].sell_orders.keys()
    #             )
    #             strawberries_price = min(
    #                 order_depths[Product.JAMS].sell_orders.keys()
    #             )
    #         elif quantity < 0 and price <= best_bid:
    #             # Sell order - trade components at their best bid prices
    #             chocolate_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
    #             strawberries_price = max(
    #                 order_depths[Product.JAMS].buy_orders.keys()
    #             )
    #         else:
    #             # The synthetic basket order does not align with the best bid or ask
    #             continue

    #         # Create orders for each component
    #         chocolate_order = Order(
    #             Product.CROISSANTS,
    #             chocolate_price,
    #             quantity * (BASKET_WEIGHTS_2[Product.CROISSANTS]),
    #         )
    #         strawberries_order = Order(
    #             Product.JAMS,
    #             strawberries_price,
    #             quantity * BASKET_WEIGHTS_2[Product.JAMS],
    #         )

    #         # Add the component orders to the respective lists
    #         component_orders[Product.CROISSANTS].append(chocolate_order)
    #         component_orders[Product.JAMS].append(strawberries_order)
    #     return component_orders
    
    # def execute_spread_orders1(
    #     self,
    #     target_position: int,
    #     basket_position: int,
    #     order_depths: Dict[str, OrderDepth],
    # ):

    #     if target_position == basket_position:
    #         return None

    #     target_quantity = abs(target_position - basket_position)
    #     basket_order_depth = order_depths[Product.GIFT_BASKET_1]
    #     synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)

    #     if target_position > basket_position:
    #         basket_ask_price = min(basket_order_depth.sell_orders.keys())
    #         basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

    #         synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
    #         synthetic_bid_volume = abs(
    #             synthetic_order_depth.buy_orders[synthetic_bid_price]
    #         )

    #         orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
    #         execute_volume = min(orderbook_volume, target_quantity)

    #         basket_orders = [
    #             Order(Product.GIFT_BASKET_1, basket_ask_price, execute_volume)
    #         ]
    #         synthetic_orders = [
    #             Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
    #         ]

    #         aggregate_orders = self.convert_synthetic_basket1_orders(
    #             synthetic_orders, order_depths
    #         )
    #         aggregate_orders[Product.GIFT_BASKET_1] = basket_orders
    #         return aggregate_orders

    #     else:
    #         basket_bid_price = max(basket_order_depth.buy_orders.keys())
    #         basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

    #         synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
    #         synthetic_ask_volume = abs(
    #             synthetic_order_depth.sell_orders[synthetic_ask_price]
    #         )

    #         orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
    #         execute_volume = min(orderbook_volume, target_quantity)

    #         basket_orders = [
    #             Order(Product.GIFT_BASKET_1, basket_bid_price, -execute_volume)
    #         ]
    #         synthetic_orders = [
    #             Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
    #         ]

    #         aggregate_orders = self.convert_synthetic_basket1_orders(
    #             synthetic_orders, order_depths
    #         )
    #         aggregate_orders[Product.GIFT_BASKET_1] = basket_orders
    #         return aggregate_orders
        
    # def execute_spread_orders2(
    #     self,
    #     target_position: int,
    #     basket_position: int,
    #     order_depths: Dict[str, OrderDepth],
    # ):

    #     if target_position == basket_position:
    #         return None

    #     target_quantity = abs(target_position - basket_position)
    #     basket_order_depth = order_depths[Product.GIFT_BASKET_2]
    #     synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

    #     if target_position > basket_position:
    #         basket_ask_price = min(basket_order_depth.sell_orders.keys())
    #         basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

    #         synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
    #         synthetic_bid_volume = abs(
    #             synthetic_order_depth.buy_orders[synthetic_bid_price]
    #         )

    #         orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
    #         execute_volume = min(orderbook_volume, target_quantity)

    #         basket_orders = [
    #             Order(Product.GIFT_BASKET_2, basket_ask_price, execute_volume)
    #         ]
    #         synthetic_orders = [
    #             Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
    #         ]

    #         aggregate_orders = self.convert_synthetic_basket2_orders(
    #             synthetic_orders, order_depths
    #         )
    #         aggregate_orders[Product.GIFT_BASKET_2] = basket_orders
    #         return aggregate_orders

    #     else:
    #         basket_bid_price = max(basket_order_depth.buy_orders.keys())
    #         basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

    #         synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
    #         synthetic_ask_volume = abs(
    #             synthetic_order_depth.sell_orders[synthetic_ask_price]
    #         )

    #         orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
    #         execute_volume = min(orderbook_volume, target_quantity)

    #         basket_orders = [
    #             Order(Product.GIFT_BASKET_2, basket_bid_price, -execute_volume)
    #         ]
    #         synthetic_orders = [
    #             Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
    #         ]

    #         aggregate_orders = self.convert_synthetic_basket2_orders(
    #             synthetic_orders, order_depths
    #         )
    #         aggregate_orders[Product.GIFT_BASKET_2] = basket_orders
    #         return aggregate_orders
        
    # def spread_orders1(
    #     self,
    #     order_depths: Dict[str, OrderDepth],
    #     product: Product,
    #     basket_position: int,
    #     spread_data: Dict[str, Any],
    # ):
    #     if Product.GIFT_BASKET_2 not in order_depths.keys():
    #         return None

    #     basket_order_depth = order_depths[Product.GIFT_BASKET_1]
    #     synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)
    #     basket_swmid = self.get_swmid(basket_order_depth)
    #     synthetic_swmid = self.get_swmid(synthetic_order_depth)
    #     spread = basket_swmid - synthetic_swmid
    #     spread_data["spread_history"].append(spread)

    #     if (
    #         len(spread_data["spread_history"])
    #         < self.params[Product.SPREAD1]["spread_std_window"]
    #     ):
    #         return None
    #     elif len(spread_data["spread_history"]) > self.params[Product.SPREAD1]["spread_std_window"]:
    #         spread_data["spread_history"].pop(0)

    #     spread_std = np.std(spread_data["spread_history"])

    #     zscore = (
    #         spread - self.params[Product.SPREAD1]["default_spread_mean"]
    #     ) / spread_std

    #     if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
    #         if basket_position != -self.params[Product.SPREAD1]["target_position"]:
    #             return self.execute_spread_orders1(
    #                 -self.params[Product.SPREAD1]["target_position"],
    #                 basket_position,
    #                 order_depths,
    #             )

    #     if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
    #         if basket_position != self.params[Product.SPREAD1]["target_position"]:
    #             return self.execute_spread_orders1(
    #                 self.params[Product.SPREAD1]["target_position"],
    #                 basket_position,
    #                 order_depths,
    #             )

    #     spread_data["prev_zscore"] = zscore
    #     return None

    # def spread_orders2(
    #     self,
    #     order_depths: Dict[str, OrderDepth],
    #     product: Product,
    #     basket_position: int,
    #     spread_data: Dict[str, Any],
    # ):
    #     if Product.GIFT_BASKET_2 not in order_depths.keys():
    #         return None

    #     # Calculate spread and z-score
    #     basket_order_depth = order_depths[Product.GIFT_BASKET_2]
    #     synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
    #     basket_swmid = self.get_swmid(basket_order_depth)
    #     synthetic_swmid = self.get_swmid(synthetic_order_depth)
    #     spread = basket_swmid - synthetic_swmid
    #     spread_data["spread_history"].append(spread)

    #     # Maintain window size
    #     if len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
    #         spread_data["spread_history"].pop(0)

    #     if len(spread_data["spread_history"]) < self.params[Product.SPREAD2]["spread_std_window"]:
    #         return None

    #     # Calculate z-score
    #     spread_std = np.std(spread_data["spread_history"])
    #     zscore = (spread - self.params[Product.SPREAD2]["default_spread_mean"]) / spread_std

    #     # Short-sell when spread is too high
    #     if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
    #         if basket_position != -self.params[Product.SPREAD2]["target_position"]:
    #             return self.execute_spread_orders2(
    #                 -self.params[Product.SPREAD2]["target_position"],
    #                 basket_position,
    #                 order_depths,
    #             )

    #     # Buy long when spread is too low
    #     if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
    #         if basket_position != self.params[Product.SPREAD2]["target_position"]:
    #             return self.execute_spread_orders2(
    #                 self.params[Product.SPREAD2]["target_position"],
    #                 basket_position,
    #                 order_depths,
    #             )

    #     # Exit if spread reverts toward mean
    #     if abs(zscore) < 0.5 and basket_position != 0:
    #         return self.execute_spread_orders2(0, basket_position, order_depths)

    #     spread_data["prev_zscore"] = zscore
    #     return None
    
    # def get_coconut_coupon_mid_price(
    #     self, coconut_coupon_order_depth: OrderDepth, traderData: Dict[str, Any]
    # ):
    #     if (
    #         len(coconut_coupon_order_depth.buy_orders) > 0
    #         and len(coconut_coupon_order_depth.sell_orders) > 0
    #     ):
    #         best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
    #         best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
    #         traderData["prev_coupon_price"] = (best_bid + best_ask) / 2
    #         return (best_bid + best_ask) / 2
    #     else:
    #         return traderData["prev_coupon_price"]

    # def delta_hedge_coconut_position(
    #     self,
    #     coconut_order_depth: OrderDepth,
    #     coconut_coupon_position: int,
    #     coconut_position: int,
    #     coconut_buy_orders: int,
    #     coconut_sell_orders: int,
    #     delta: float,
    # ) -> List[Order]:

    #     target_coconut_position = -int(delta * coconut_coupon_position)
    #     hedge_quantity = target_coconut_position - (
    #         coconut_position + coconut_buy_orders - coconut_sell_orders
    #     )

    #     orders: List[Order] = []
    #     if hedge_quantity > 0:
    #         # Buy COCONUT
    #         best_ask = min(coconut_order_depth.sell_orders.keys())
    #         quantity = min(
    #             abs(hedge_quantity), -coconut_order_depth.sell_orders[best_ask]
    #         )
    #         quantity = min(
    #             quantity,
    #             self.LIMIT[Product.COCONUT] - (coconut_position + coconut_buy_orders),
    #         )
    #         if quantity > 0:
    #             orders.append(Order(Product.COCONUT, best_ask, quantity))
    #     elif hedge_quantity < 0:
    #         # Sell COCONUT
    #         best_bid = max(coconut_order_depth.buy_orders.keys())
    #         quantity = min(
    #             abs(hedge_quantity), coconut_order_depth.buy_orders[best_bid]
    #         )
    #         quantity = min(
    #             quantity,
    #             self.LIMIT[Product.COCONUT] + (coconut_position - coconut_sell_orders),
    #         )
    #         if quantity > 0:
    #             orders.append(Order(Product.COCONUT, best_bid, -quantity))

    #     return orders

    # def delta_hedge_coconut_coupon_orders(
    #     self,
    #     coconut_order_depth: OrderDepth,
    #     coconut_coupon_orders: List[Order],
    #     coconut_position: int,
    #     coconut_buy_orders: int,
    #     coconut_sell_orders: int,
    #     delta: float,
    # ) -> List[Order]:
    #     if len(coconut_coupon_orders) == 0:
    #         return None

    #     net_coconut_coupon_quantity = sum(
    #         order.quantity for order in coconut_coupon_orders
    #     )
    #     target_coconut_quantity = -int(delta * net_coconut_coupon_quantity)

    #     orders: List[Order] = []
    #     if target_coconut_quantity > 0:
    #         # Buy COCONUT
    #         best_ask = min(coconut_order_depth.sell_orders.keys())
    #         quantity = min(
    #             abs(target_coconut_quantity), -coconut_order_depth.sell_orders[best_ask]
    #         )
    #         quantity = min(
    #             quantity,
    #             self.LIMIT[Product.COCONUT] - (coconut_position + coconut_buy_orders),
    #         )
    #         if quantity > 0:
    #             orders.append(Order(Product.COCONUT, best_ask, quantity))
    #     elif target_coconut_quantity < 0:
    #         # Sell COCONUT
    #         best_bid = max(coconut_order_depth.buy_orders.keys())
    #         quantity = min(
    #             abs(target_coconut_quantity), coconut_order_depth.buy_orders[best_bid]
    #         )
    #         quantity = min(
    #             quantity,
    #             self.LIMIT[Product.COCONUT] + (coconut_position - coconut_sell_orders),
    #         )
    #         if quantity > 0:
    #             orders.append(Order(Product.COCONUT, best_bid, -quantity))

    #     return orders

    # def coconut_hedge_orders(
    #     self,
    #     coconut_order_depth: OrderDepth,
    #     coconut_coupon_order_depth: OrderDepth,
    #     coconut_coupon_orders: List[Order],
    #     coconut_position: int,
    #     coconut_coupon_position: int,
    #     delta: float,
    # ) -> List[Order]:
    #     if coconut_coupon_orders == None or len(coconut_coupon_orders) == 0:
    #         coconut_coupon_position_after_trade = coconut_coupon_position
    #     else:
    #         coconut_coupon_position_after_trade = coconut_coupon_position + sum(
    #             order.quantity for order in coconut_coupon_orders
    #         )

    #     target_coconut_position = -delta * coconut_coupon_position_after_trade

    #     if target_coconut_position == coconut_position:
    #         return None

    #     target_coconut_quantity = target_coconut_position - coconut_position

    #     orders: List[Order] = []
    #     if target_coconut_quantity > 0:
    #         # Buy COCONUT
    #         best_ask = min(coconut_order_depth.sell_orders.keys())
    #         quantity = min(
    #             abs(target_coconut_quantity),
    #             self.LIMIT[Product.COCONUT] - coconut_position,
    #         )
    #         if quantity > 0:
    #             orders.append(Order(Product.COCONUT, best_ask, round(quantity)))

    #     elif target_coconut_quantity < 0:
    #         # Sell COCONUT
    #         best_bid = max(coconut_order_depth.buy_orders.keys())
    #         quantity = min(
    #             abs(target_coconut_quantity),
    #             self.LIMIT[Product.COCONUT] + coconut_position,
    #         )
    #         if quantity > 0:
    #             orders.append(Order(Product.COCONUT, best_bid, -round(quantity)))

    #     return orders

    # def coconut_coupon_orders(
    #     self,
    #     coconut_coupon_order_depth: OrderDepth,
    #     coconut_coupon_position: int,
    #     traderData: Dict[str, Any],
    #     volatility: float,
    # ) -> List[Order]:
    #     traderData["past_coupon_vol"].append(volatility)
    #     if (
    #         len(traderData["past_coupon_vol"])
    #         < self.params[Product.COCONUT_COUPON]["std_window"]
    #     ):
    #         return None, None

    #     if (
    #         len(traderData["past_coupon_vol"])
    #         > self.params[Product.COCONUT_COUPON]["std_window"]
    #     ):
    #         traderData["past_coupon_vol"].pop(0)

    #     vol_z_score = (
    #         volatility - self.params[Product.COCONUT_COUPON]["mean_volatility"]
    #     ) / np.std(traderData["past_coupon_vol"])
    #     # print(f"vol_z_score: {vol_z_score}")
    #     # print(f"zscore_threshold: {self.params[Product.COCONUT_COUPON]['zscore_threshold']}")
    #     if vol_z_score >= self.params[Product.COCONUT_COUPON]["zscore_threshold"]:
    #         if coconut_coupon_position != -self.LIMIT[Product.COCONUT_COUPON]:
    #             target_coconut_coupon_position = -self.LIMIT[Product.COCONUT_COUPON]
    #             if len(coconut_coupon_order_depth.buy_orders) > 0:
    #                 best_bid = max(coconut_coupon_order_depth.buy_orders.keys())
    #                 target_quantity = abs(
    #                     target_coconut_coupon_position - coconut_coupon_position
    #                 )
    #                 quantity = min(
    #                     target_quantity,
    #                     abs(coconut_coupon_order_depth.buy_orders[best_bid]),
    #                 )
    #                 quote_quantity = target_quantity - quantity
    #                 if quote_quantity == 0:
    #                     return [Order(Product.COCONUT_COUPON, best_bid, -quantity)], []
    #                 else:
    #                     return [Order(Product.COCONUT_COUPON, best_bid, -quantity)], [
    #                         Order(Product.COCONUT_COUPON, best_bid, -quote_quantity)
    #                     ]

    #     elif vol_z_score <= -self.params[Product.COCONUT_COUPON]["zscore_threshold"]:
    #         if coconut_coupon_position != self.LIMIT[Product.COCONUT_COUPON]:
    #             target_coconut_coupon_position = self.LIMIT[Product.COCONUT_COUPON]
    #             if len(coconut_coupon_order_depth.sell_orders) > 0:
    #                 best_ask = min(coconut_coupon_order_depth.sell_orders.keys())
    #                 target_quantity = abs(
    #                     target_coconut_coupon_position - coconut_coupon_position
    #                 )
    #                 quantity = min(
    #                     target_quantity,
    #                     abs(coconut_coupon_order_depth.sell_orders[best_ask]),
    #                 )
    #                 quote_quantity = target_quantity - quantity
    #                 if quote_quantity == 0:
    #                     return [Order(Product.COCONUT_COUPON, best_ask, quantity)], []
    #                 else:
    #                     return [Order(Product.COCONUT_COUPON, best_ask, quantity)], [
    #                         Order(Product.COCONUT_COUPON, best_ask, quote_quantity)
    #                     ]

    #     return None, None

    # def get_past_returns(
    #     self,
    #     traderObject: Dict[str, Any],
    #     order_depths: Dict[str, OrderDepth],
    #     timeframes: Dict[str, int],
    # ):
    #     returns_dict = {}

    #     for symbol, timeframe in timeframes.items():
    #         traderObject_key = f"{symbol}_price_history"
    #         if traderObject_key not in traderObject:
    #             traderObject[traderObject_key] = []

    #         price_history = traderObject[traderObject_key]

    #         if symbol in order_depths:
    #             order_depth = order_depths[symbol]
    #             if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
    #                 current_price = (
    #                     max(order_depth.buy_orders.keys())
    #                     + min(order_depth.sell_orders.keys())
    #                 ) / 2
    #             else:
    #                 if len(price_history) > 0:
    #                     current_price = float(price_history[-1])
    #                 else:
    #                     returns_dict[symbol] = None
    #                     continue
    #         else:
    #             if len(price_history) > 0:
    #                 current_price = float(price_history[-1])
    #             else:
    #                 returns_dict[symbol] = None
    #                 continue

    #         price_history.append(
    #             f"{current_price:.1f}"
    #         )  # Convert float to string with 1 decimal place

    #         if len(price_history) > timeframe:
    #             price_history.pop(0)

    #         if len(price_history) == timeframe:
    #             past_price = float(price_history[0])  # Convert string back to float
    #             returns = (current_price - past_price) / past_price
    #             returns_dict[symbol] = returns
    #         else:
    #             returns_dict[symbol] = None

    #     return returns_dict
    
        
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        # if Product.AMETHYSTS in self.params and Product.AMETHYSTS in state.order_depths:
        #     amethyst_position = (
        #         state.position[Product.AMETHYSTS]
        #         if Product.AMETHYSTS in state.position
        #         else 0
        #     )
        #     amethyst_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.AMETHYSTS,
        #             state.order_depths[Product.AMETHYSTS],
        #             self.params[Product.AMETHYSTS]["fair_value"],
        #             self.params[Product.AMETHYSTS]["take_width"],
        #             amethyst_position,
        #         )
        #     )
        #     amethyst_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.AMETHYSTS,
        #             state.order_depths[Product.AMETHYSTS],
        #             self.params[Product.AMETHYSTS]["fair_value"],
        #             self.params[Product.AMETHYSTS]["clear_width"],
        #             amethyst_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     amethyst_make_orders, _, _ = self.make_amethyst_orders(
        #         state.order_depths[Product.AMETHYSTS],
        #         self.params[Product.AMETHYSTS]["fair_value"],
        #         amethyst_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.AMETHYSTS]["volume_limit"],
        #     )
        #     result[Product.AMETHYSTS] = (
        #         amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
        #     )

        # if Product.STARFRUIT in self.params and Product.STARFRUIT in state.order_depths:
        #     starfruit_position = (
        #         state.position[Product.STARFRUIT]
        #         if Product.STARFRUIT in state.position
        #         else 0
        #     )
        #     starfruit_fair_value = self.starfruit_fair_value(
        #         state.order_depths[Product.STARFRUIT], traderObject
        #     )
        #     starfruit_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.STARFRUIT,
        #             state.order_depths[Product.STARFRUIT],
        #             starfruit_fair_value,
        #             self.params[Product.STARFRUIT]["take_width"],
        #             starfruit_position,
        #             self.params[Product.STARFRUIT]["prevent_adverse"],
        #             self.params[Product.STARFRUIT]["adverse_volume"],
        #         )
        #     )
        #     starfruit_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.STARFRUIT,
        #             state.order_depths[Product.STARFRUIT],
        #             starfruit_fair_value,
        #             self.params[Product.STARFRUIT]["clear_width"],
        #             starfruit_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     starfruit_make_orders, _, _ = self.make_starfruit_orders(
        #         state.order_depths[Product.STARFRUIT],
        #         starfruit_fair_value,
        #         self.params[Product.STARFRUIT]["starfruit_min_edge"],
        #         starfruit_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #     )
        #     result[Product.STARFRUIT] = (
        #         starfruit_take_orders + starfruit_clear_orders + starfruit_make_orders
        #     )

        if Product.MAGNIFICENT_MACARONS in state.order_depths:
            depth: OrderDepth = state.order_depths[Product.MAGNIFICENT_MACARONS]
            if depth.buy_orders and depth.sell_orders:
                # Level-1 book
                best_bid = max(depth.buy_orders.keys())
                best_ask = min(depth.sell_orders.keys())
                bid_vol = abs(depth.buy_orders[best_bid])
                ask_vol = abs(depth.sell_orders[best_ask])
                mid = (best_bid * ask_vol + best_ask * bid_vol) / (ask_vol + bid_vol)

                # Maintain history
                hist = traderData.get("macaron_hist", [])
                hist.append(mid)
                if len(hist) > self.WINDOW:
                    hist.pop(0)
                traderData["macaron_hist"] = hist

                # Generate orders once we have full window
                orders: List[Order] = []
                pos = state.position.get(Product.MAGNIFICENT_MACARONS, 0)
                limit = self.LIMIT[Product.MAGNIFICENT_MACARONS]

                if len(hist) == 20:
                    # compute mean and std over previous WINDOW-1 ticks
                    prev = hist[:-1]
                    mean = sum(prev) / len(prev)
                    variance = sum((x - mean)**2 for x in prev) / len(prev)
                    std = variance**0.5
                    # compute z-score for current mid
                    z = (mid - mean) / std if std > 0 else 0
                    # mean-reversion thresholds: z > +1 => sell, z < -1 => buy
                    if z > 1.0 and pos > -limit:
                        qty = min(10, limit + pos)
                        orders.append(Order(Product.MAGNIFICENT_MACARONS, best_ask - 1, -qty))
                    elif z < -1.0 and pos < limit:
                        qty = min(10, limit - pos)
                        orders.append(Order(Product.MAGNIFICENT_MACARONS, best_bid + 1, qty))

                if orders:
                    result[Product.MAGNIFICENT_MACARONS] = orders
        # if Product.SPREAD1 not in traderObject:
        #     traderObject[Product.SPREAD1] = {
        #         "spread_history": [],
        #         "prev_zscore": 0,
        #         "clear_flag": False,
        #         "curr_avg": 0,
        #     }

        # basket_position1 = (
        #     state.position[Product.GIFT_BASKET_1]
        #     if Product.GIFT_BASKET_1 in state.position
        #     else 0
        # )
        # if Product.SPREAD2 not in traderObject:
        #     traderObject[Product.SPREAD2] = {
        #         "spread_history": [],
        #         "prev_zscore": 0,
        #         "clear_flag": False,
        #         "curr_avg": 0,
        #     }

        # basket_position2 = (
        #     state.position[Product.GIFT_BASKET_2]
        #     if Product.GIFT_BASKET_2 in state.position
        #     else 0
        # )
        # spread_orders1 = self.spread_orders1(
        #     state.order_depths,
        #     Product.GIFT_BASKET_1,
        #     basket_position1,
        #     traderObject[Product.SPREAD1],
        # )
        # if spread_orders1 != None:
        #     result[Product.CROISSANTS] = spread_orders1[Product.CROISSANTS]
        #     result[Product.JAMS] = spread_orders1[Product.JAMS]
        #     result[Product.DJEMBE] = spread_orders1[Product.DJEMBE]
        #     result[Product.GIFT_BASKET_1] = spread_orders1[Product.GIFT_BASKET_1]

        # spread_orders2 = self.spread_orders2(
        #     state.order_depths,
        #     Product.GIFT_BASKET_2,
        #     basket_position2,
        #     traderObject[Product.SPREAD2],
        # )
        # if spread_orders2 != None:
        #     result[Product.CROISSANTS] = spread_orders2[Product.CROISSANTS]
        #     result[Product.JAMS] = spread_orders2[Product.JAMS]
        #     result[Product.GIFT_BASKET_2] = spread_orders2[Product.GIFT_BASKET_2]
        
        # if Product.COCONUT_COUPON not in traderObject:
        #     traderObject[Product.COCONUT_COUPON] = {
        #         "prev_coupon_price": 0,
        #         "past_coupon_vol": [],
        #     }

        # if (
        #     Product.COCONUT_COUPON in self.params
        #     and Product.COCONUT_COUPON in state.order_depths
        # ):
        #     coconut_coupon_position = (
        #         state.position[Product.COCONUT_COUPON]
        #         if Product.COCONUT_COUPON in state.position
        #         else 0
        #     )

        #     coconut_position = (
        #         state.position[Product.COCONUT]
        #         if Product.COCONUT in state.position
        #         else 0
        #     )
        #     # print(f"coconut_coupon_position: {coconut_coupon_position}")
        #     # print(f"coconut_position: {coconut_position}")
        #     coconut_order_depth = state.order_depths[Product.COCONUT]
        #     coconut_coupon_order_depth = state.order_depths[Product.COCONUT_COUPON]
        #     coconut_mid_price = (
        #         min(coconut_order_depth.buy_orders.keys())
        #         + max(coconut_order_depth.sell_orders.keys())
        #     ) / 2
        #     coconut_coupon_mid_price = self.get_coconut_coupon_mid_price(
        #         coconut_coupon_order_depth, traderObject[Product.COCONUT_COUPON]
        #     )
        #     tte = (
        #         self.params[Product.COCONUT_COUPON]["starting_time_to_expiry"]
        #         - (state.timestamp) / 1000000 / 250
        #     )
        #     volatility = BlackScholes.implied_volatility(
        #         coconut_coupon_mid_price,
        #         coconut_mid_price,
        #         self.params[Product.COCONUT_COUPON]["strike"],
        #         tte,
        #     )
        #     delta = BlackScholes.delta(
        #         coconut_mid_price,
        #         self.params[Product.COCONUT_COUPON]["strike"],
        #         tte,
        #         volatility,
        #     )

        #     coconut_coupon_take_orders, coconut_coupon_make_orders = (
        #         self.coconut_coupon_orders(
        #             state.order_depths[Product.COCONUT_COUPON],
        #             coconut_coupon_position,
        #             traderObject[Product.COCONUT_COUPON],
        #             volatility,
        #         )
        #     )

        #     coconut_orders = self.coconut_hedge_orders(
        #         state.order_depths[Product.COCONUT],
        #         state.order_depths[Product.COCONUT_COUPON],
        #         coconut_coupon_take_orders,
        #         coconut_position,
        #         coconut_coupon_position,
        #         delta,
        #     )

        #     if coconut_coupon_take_orders != None or coconut_coupon_make_orders != None:
        #         result[Product.COCONUT_COUPON] = (
        #             coconut_coupon_take_orders + coconut_coupon_make_orders
        #         )
        #         # print(f"COCONUT_COUPON: {result[Product.COCONUT_COUPON]}")

        #     if coconut_orders != None:
        #         result[Product.COCONUT] = coconut_orders
                # print(f"COCONUT: {result[Product.COCONUT]}")
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData