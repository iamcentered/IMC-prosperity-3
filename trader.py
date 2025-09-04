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
    VOLCANIC_ROCK = "VOLCANIC_ROCK"
    

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
    Product.VOLCANIC_ROCK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.229,
        "starfruit_min_edge": 2,
    },
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
            Product.VOLCANIC_ROCK: 400,
        }
    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int):
        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if best_ask <= fair_value - take_width:
                quantity = min(
                    best_ask_amount, position_limit - position
                )  # max amt to buy
                if quantity > 0:
                    orders.append(Order(product, best_ask, quantity))
                    buy_order_volume += quantity
                    order_depth.sell_orders[best_ask] += quantity
                    if order_depth.sell_orders[best_ask] == 0:
                        del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                quantity = min(
                    best_bid_amount, position_limit + position
                )  # should be the max we can sell
                if quantity > 0:
                    orders.append(Order(product, best_bid, -1 * quantity))
                    sell_order_volume += quantity
                    order_depth.buy_orders[best_bid] -= quantity
                    if order_depth.buy_orders[best_bid] == 0:
                        del order_depth.buy_orders[best_bid]
        return buy_order_volume, sell_order_volume

    def take_best_orders_with_adverse(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        adverse_volume: int,
    ) -> (int, int):

        position_limit = self.LIMIT[product]
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            if abs(best_ask_amount) <= adverse_volume:
                if best_ask <= fair_value - take_width:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]
            if abs(best_bid_amount) <= adverse_volume:
                if best_bid >= fair_value + take_width:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume
    
    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )
            clear_quantity = min(clear_quantity, position_after_take)
            sent_quantity = min(sell_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def volcanic_rock_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("volcanic_rock_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["volcanic_rock_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("volcanic_rock_last_price", None) != None:
                last_price = traderObject["volcanic_rock_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.VOLCANIC_ROCK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["volcanic_rock_last_price"] = mmmid_price
            return fair
        return None
    
    def starfruit_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.VOLCANIC_ROCK]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("starfruit_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["starfruit_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("starfruit_last_price", None) != None:
                last_price = traderObject["starfruit_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.VOLCANIC_ROCK]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["starfruit_last_price"] = mmmid_price
            return fair
        return None
    def make_amethyst_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        volume_limit: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        baaf = min(
            [
                price
                for price in order_depth.sell_orders.keys()
                if price > fair_value + 1
            ]
        )
        bbbf = max(
            [price for price in order_depth.buy_orders.keys() if price < fair_value - 1]
        )

        if baaf <= fair_value + 2:
            if position <= volume_limit:
                baaf = fair_value + 3  # still want edge 2 if position is not a concern

        if bbbf >= fair_value - 2:
            if position >= -volume_limit:
                bbbf = fair_value - 3  # still want edge 2 if position is not a concern

        buy_order_volume, sell_order_volume = self.market_make(
            Product.AMETHYSTS,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    # def squid_ink_ma_crossover(self, order_depth: OrderDepth, traderObject) -> float:
    #     if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
    #         best_ask = min(order_depth.sell_orders.keys())
    #         best_bid = max(order_depth.buy_orders.keys())
    #         mid_price = (best_ask + best_bid) / 2

    #         # Maintain price history for 1000 ticks
    #         price_history = traderObject.get("squid_ink_price_history", [])
    #         price_history.append(mid_price)

    #         # Keep only last 1000 prices
    #         if len(price_history) > 1000:
    #             price_history.pop(0)
    #         traderObject["squid_ink_price_history"] = price_history

    #         # Initialize default signal
    #         traderObject["squid_ink_signal"] = traderObject.get("squid_ink_signal", "HOLD")

    #         # Calculate indicators
    #         if len(price_history) >= 1000:
    #             ma_1000 = sum(price_history) / 1000
    #             prev_ma = traderObject.get("squid_ink_prev_ma", ma_1000)
    #             prev_price = traderObject.get("squid_ink_prev_price", mid_price)

    #             # Detect crossover events
    #             if prev_price < prev_ma and mid_price > ma_1000:
    #                 traderObject["squid_ink_signal"] = "BUY"
    #             elif prev_price > prev_ma and mid_price < ma_1000:
    #                 traderObject["squid_ink_signal"] = "SELL"

    #             # Update previous values
    #             traderObject["squid_ink_prev_ma"] = ma_1000
    #             traderObject["squid_ink_prev_price"] = mid_price

    #         return mid_price
    #     return None
    def squidink_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            mid_price = (best_ask + best_bid) / 2

            # Maintain price history for 1000 ticks
            price_history = traderObject.get("squidink_price_history", [])
            price_history.append(mid_price)

            # Keep only last 1000 prices
            if len(price_history) > 1000:
                price_history.pop(0)
            traderObject["squidink_price_history"] = price_history

            # Initialize default signal
            traderObject["squidink_signal"] = traderObject.get("squidink_signal", "HOLD")

            # Calculate indicators
            if len(price_history) >= 1000:
                ma_1000 = sum(price_history) / 1000
                prev_ma = traderObject.get("squidink_prev_ma", ma_1000)
                prev_price = traderObject.get("squidink_prev_price", mid_price)

                # Detect crossover events
                if prev_price < prev_ma and mid_price > ma_1000:
                    traderObject["squidink_signal"] = "BUY"
                elif prev_price > prev_ma and mid_price < ma_1000:
                    traderObject["squidink_signal"] = "SELL"

                # Update previous values
                traderObject["squidink_prev_ma"] = ma_1000
                traderObject["squidink_prev_price"] = mid_price

            return ma_1000  # Return MA as fair value
        return None

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume = 0
        sell_order_volume = 0

        if prevent_adverse:
            buy_order_volume, sell_order_volume = self.take_best_orders_with_adverse(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
                adverse_volume,
            )
        else:
            buy_order_volume, sell_order_volume = self.take_best_orders(
                product,
                fair_value,
                take_width,
                orders,
                order_depth,
                position,
                buy_order_volume,
                sell_order_volume,
            )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def make_starfruit_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.STARFRUIT,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def make_volcanic_rock_orders(
        self,
        order_depth: OrderDepth,
        fair_value: float,
        min_edge: float,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int):
        orders: List[Order] = []
        aaf = [
            price
            for price in order_depth.sell_orders.keys()
            if price >= round(fair_value + min_edge)
        ]
        bbf = [
            price
            for price in order_depth.buy_orders.keys()
            if price <= round(fair_value - min_edge)
        ]
        baaf = min(aaf) if len(aaf) > 0 else round(fair_value + min_edge)
        bbbf = max(bbf) if len(bbf) > 0 else round(fair_value - min_edge)
        buy_order_volume, sell_order_volume = self.market_make(
            Product.VOLCANIC_ROCK,
            orders,
            bbbf + 1,
            baaf - 1,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume
    
    def get_swmid(self, order_depth) -> float:
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        best_bid_vol = abs(order_depth.buy_orders[best_bid])
        best_ask_vol = abs(order_depth.sell_orders[best_ask])
        return (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
            best_bid_vol + best_ask_vol
        )
    
    def get_synthetic_basket1_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CHOCOLATE_PER_BASKET = BASKET_WEIGHTS_1[Product.CROISSANTS]
        STRAWBERRIES_PER_BASKET = BASKET_WEIGHTS_1[Product.JAMS]
        ROSES_PER_BASKET = BASKET_WEIGHTS_1[Product.DJEMBE]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        chocolate_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        chocolate_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        strawberries_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        strawberries_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        roses_best_bid = (
            max(order_depths[Product.DJEMBE].buy_orders.keys())
            if order_depths[Product.DJEMBE].buy_orders
            else 0
        )
        roses_best_ask = (
            min(order_depths[Product.DJEMBE].sell_orders.keys())
            if order_depths[Product.DJEMBE].sell_orders
            else float("inf")
        )

        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            chocolate_best_bid * CHOCOLATE_PER_BASKET
            + strawberries_best_bid * STRAWBERRIES_PER_BASKET
            + roses_best_bid * ROSES_PER_BASKET
        )
        implied_ask = (
            chocolate_best_ask * CHOCOLATE_PER_BASKET
            + strawberries_best_ask * STRAWBERRIES_PER_BASKET
            + roses_best_ask * ROSES_PER_BASKET
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            chocolate_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[chocolate_best_bid]
                // CHOCOLATE_PER_BASKET
            )
            strawberries_bid_volume = (
                order_depths[Product.JAMS].buy_orders[strawberries_best_bid]
                // STRAWBERRIES_PER_BASKET
            )
            roses_bid_volume = (
                order_depths[Product.DJEMBE].buy_orders[roses_best_bid]
                // ROSES_PER_BASKET
            )
            implied_bid_volume = min(
                chocolate_bid_volume, strawberries_bid_volume, roses_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            chocolate_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[chocolate_best_ask]
                // CHOCOLATE_PER_BASKET
            )
            strawberries_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[strawberries_best_ask]
                // STRAWBERRIES_PER_BASKET
            )
            roses_ask_volume = (
                -order_depths[Product.DJEMBE].sell_orders[roses_best_ask]
                // ROSES_PER_BASKET
            )
            implied_ask_volume = min(
                chocolate_ask_volume, strawberries_ask_volume, roses_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price
    
    def get_synthetic_basket2_order_depth(
        self, order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        # Constants
        CHOCOLATE_PER_BASKET2 = BASKET_WEIGHTS_2[Product.CROISSANTS]
        STRAWBERRIES_PER_BASKET2 = BASKET_WEIGHTS_2[Product.JAMS]

        # Initialize the synthetic basket order depth
        synthetic_order_price = OrderDepth()

        # Calculate the best bid and ask for each component
        chocolate_best_bid = (
            max(order_depths[Product.CROISSANTS].buy_orders.keys())
            if order_depths[Product.CROISSANTS].buy_orders
            else 0
        )
        chocolate_best_ask = (
            min(order_depths[Product.CROISSANTS].sell_orders.keys())
            if order_depths[Product.CROISSANTS].sell_orders
            else float("inf")
        )
        strawberries_best_bid = (
            max(order_depths[Product.JAMS].buy_orders.keys())
            if order_depths[Product.JAMS].buy_orders
            else 0
        )
        strawberries_best_ask = (
            min(order_depths[Product.JAMS].sell_orders.keys())
            if order_depths[Product.JAMS].sell_orders
            else float("inf")
        )
        # Calculate the implied bid and ask for the synthetic basket
        implied_bid = (
            chocolate_best_bid * CHOCOLATE_PER_BASKET2
            + strawberries_best_bid * STRAWBERRIES_PER_BASKET2
        )
        implied_ask = (
            chocolate_best_ask * CHOCOLATE_PER_BASKET2
            + strawberries_best_ask * STRAWBERRIES_PER_BASKET2
        )

        # Calculate the maximum number of synthetic baskets available at the implied bid and ask
        if implied_bid > 0:
            chocolate_bid_volume = (
                order_depths[Product.CROISSANTS].buy_orders[chocolate_best_bid]
                // CHOCOLATE_PER_BASKET2
            )
            strawberries_bid_volume = (
                order_depths[Product.JAMS].buy_orders[strawberries_best_bid]
                // STRAWBERRIES_PER_BASKET2
            )
            implied_bid_volume = min(
                chocolate_bid_volume, strawberries_bid_volume
            )
            synthetic_order_price.buy_orders[implied_bid] = implied_bid_volume

        if implied_ask < float("inf"):
            chocolate_ask_volume = (
                -order_depths[Product.CROISSANTS].sell_orders[chocolate_best_ask]
                // CHOCOLATE_PER_BASKET2
            )
            strawberries_ask_volume = (
                -order_depths[Product.JAMS].sell_orders[strawberries_best_ask]
                // STRAWBERRIES_PER_BASKET2
            )
            implied_ask_volume = min(
                chocolate_ask_volume, strawberries_ask_volume
            )
            synthetic_order_price.sell_orders[implied_ask] = -implied_ask_volume

        return synthetic_order_price

    def convert_synthetic_basket1_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
            Product.DJEMBE: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket1_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                chocolate_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                strawberries_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
                roses_price = min(order_depths[Product.DJEMBE].sell_orders.keys())
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                chocolate_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                strawberries_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
                roses_price = max(order_depths[Product.DJEMBE].buy_orders.keys())
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            chocolate_order = Order(
                Product.CROISSANTS,
                chocolate_price,
                quantity * (BASKET_WEIGHTS_1[Product.CROISSANTS]),
            )
            strawberries_order = Order(
                Product.JAMS,
                strawberries_price,
                quantity * BASKET_WEIGHTS_1[Product.JAMS],
            )
            roses_order = Order(
                Product.DJEMBE, roses_price, quantity * BASKET_WEIGHTS_1[Product.DJEMBE]
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(chocolate_order)
            component_orders[Product.JAMS].append(strawberries_order)
            component_orders[Product.DJEMBE].append(roses_order)

        return component_orders
    
    def convert_synthetic_basket2_orders(
        self, synthetic_orders: List[Order], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        # Initialize the dictionary to store component orders
        component_orders = {
            Product.CROISSANTS: [],
            Product.JAMS: [],
        }

        # Get the best bid and ask for the synthetic basket
        synthetic_basket_order_depth = self.get_synthetic_basket2_order_depth(
            order_depths
        )
        best_bid = (
            max(synthetic_basket_order_depth.buy_orders.keys())
            if synthetic_basket_order_depth.buy_orders
            else 0
        )
        best_ask = (
            min(synthetic_basket_order_depth.sell_orders.keys())
            if synthetic_basket_order_depth.sell_orders
            else float("inf")
        )

        # Iterate through each synthetic basket order
        for order in synthetic_orders:
            # Extract the price and quantity from the synthetic basket order
            price = order.price
            quantity = order.quantity

            # Check if the synthetic basket order aligns with the best bid or ask
            if quantity > 0 and price >= best_ask:
                # Buy order - trade components at their best ask prices
                chocolate_price = min(
                    order_depths[Product.CROISSANTS].sell_orders.keys()
                )
                strawberries_price = min(
                    order_depths[Product.JAMS].sell_orders.keys()
                )
            elif quantity < 0 and price <= best_bid:
                # Sell order - trade components at their best bid prices
                chocolate_price = max(order_depths[Product.CROISSANTS].buy_orders.keys())
                strawberries_price = max(
                    order_depths[Product.JAMS].buy_orders.keys()
                )
            else:
                # The synthetic basket order does not align with the best bid or ask
                continue

            # Create orders for each component
            chocolate_order = Order(
                Product.CROISSANTS,
                chocolate_price,
                quantity * (BASKET_WEIGHTS_2[Product.CROISSANTS]),
            )
            strawberries_order = Order(
                Product.JAMS,
                strawberries_price,
                quantity * BASKET_WEIGHTS_2[Product.JAMS],
            )

            # Add the component orders to the respective lists
            component_orders[Product.CROISSANTS].append(chocolate_order)
            component_orders[Product.JAMS].append(strawberries_order)
        return component_orders
    
    def execute_spread_orders1(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET_1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET_1, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket1_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET_1] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET_1, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket1_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET_1] = basket_orders
            return aggregate_orders
        
    def execute_spread_orders2(
        self,
        target_position: int,
        basket_position: int,
        order_depths: Dict[str, OrderDepth],
    ):

        if target_position == basket_position:
            return None

        target_quantity = abs(target_position - basket_position)
        basket_order_depth = order_depths[Product.GIFT_BASKET_2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)

        if target_position > basket_position:
            basket_ask_price = min(basket_order_depth.sell_orders.keys())
            basket_ask_volume = abs(basket_order_depth.sell_orders[basket_ask_price])

            synthetic_bid_price = max(synthetic_order_depth.buy_orders.keys())
            synthetic_bid_volume = abs(
                synthetic_order_depth.buy_orders[synthetic_bid_price]
            )

            orderbook_volume = min(basket_ask_volume, synthetic_bid_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET_2, basket_ask_price, execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_bid_price, -execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET_2] = basket_orders
            return aggregate_orders

        else:
            basket_bid_price = max(basket_order_depth.buy_orders.keys())
            basket_bid_volume = abs(basket_order_depth.buy_orders[basket_bid_price])

            synthetic_ask_price = min(synthetic_order_depth.sell_orders.keys())
            synthetic_ask_volume = abs(
                synthetic_order_depth.sell_orders[synthetic_ask_price]
            )

            orderbook_volume = min(basket_bid_volume, synthetic_ask_volume)
            execute_volume = min(orderbook_volume, target_quantity)

            basket_orders = [
                Order(Product.GIFT_BASKET_2, basket_bid_price, -execute_volume)
            ]
            synthetic_orders = [
                Order(Product.SYNTHETIC, synthetic_ask_price, execute_volume)
            ]

            aggregate_orders = self.convert_synthetic_basket2_orders(
                synthetic_orders, order_depths
            )
            aggregate_orders[Product.GIFT_BASKET_2] = basket_orders
            return aggregate_orders
        
    def spread_orders1(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.GIFT_BASKET_2 not in order_depths.keys():
            return None

        basket_order_depth = order_depths[Product.GIFT_BASKET_1]
        synthetic_order_depth = self.get_synthetic_basket1_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        if (
            len(spread_data["spread_history"])
            < self.params[Product.SPREAD1]["spread_std_window"]
        ):
            return None
        elif len(spread_data["spread_history"]) > self.params[Product.SPREAD1]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        spread_std = np.std(spread_data["spread_history"])

        zscore = (
            spread - self.params[Product.SPREAD1]["default_spread_mean"]
        ) / spread_std

        if zscore >= self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread_orders1(
                    -self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        if zscore <= -self.params[Product.SPREAD1]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD1]["target_position"]:
                return self.execute_spread_orders1(
                    self.params[Product.SPREAD1]["target_position"],
                    basket_position,
                    order_depths,
                )

        spread_data["prev_zscore"] = zscore
        return None

    def spread_orders2(
        self,
        order_depths: Dict[str, OrderDepth],
        product: Product,
        basket_position: int,
        spread_data: Dict[str, Any],
    ):
        if Product.GIFT_BASKET_2 not in order_depths.keys():
            return None

        # Calculate spread and z-score
        basket_order_depth = order_depths[Product.GIFT_BASKET_2]
        synthetic_order_depth = self.get_synthetic_basket2_order_depth(order_depths)
        basket_swmid = self.get_swmid(basket_order_depth)
        synthetic_swmid = self.get_swmid(synthetic_order_depth)
        spread = basket_swmid - synthetic_swmid
        spread_data["spread_history"].append(spread)

        # Maintain window size
        if len(spread_data["spread_history"]) > self.params[Product.SPREAD2]["spread_std_window"]:
            spread_data["spread_history"].pop(0)

        if len(spread_data["spread_history"]) < self.params[Product.SPREAD2]["spread_std_window"]:
            return None

        # Calculate z-score
        spread_std = np.std(spread_data["spread_history"])
        zscore = (spread - self.params[Product.SPREAD2]["default_spread_mean"]) / spread_std

        # Short-sell when spread is too high
        if zscore >= self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != -self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders2(
                    -self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        # Buy long when spread is too low
        if zscore <= -self.params[Product.SPREAD2]["zscore_threshold"]:
            if basket_position != self.params[Product.SPREAD2]["target_position"]:
                return self.execute_spread_orders2(
                    self.params[Product.SPREAD2]["target_position"],
                    basket_position,
                    order_depths,
                )

        # Exit if spread reverts toward mean
        if abs(zscore) < 0.5 and basket_position != 0:
            return self.execute_spread_orders2(0, basket_position, order_depths)

        spread_data["prev_zscore"] = zscore
        return None
    
        
    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}
        conversions = 0

        if Product.AMETHYSTS in self.params and Product.AMETHYSTS in state.order_depths:
            amethyst_position = (
                state.position[Product.AMETHYSTS]
                if Product.AMETHYSTS in state.position
                else 0
            )
            amethyst_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.AMETHYSTS,
                    state.order_depths[Product.AMETHYSTS],
                    self.params[Product.AMETHYSTS]["fair_value"],
                    self.params[Product.AMETHYSTS]["take_width"],
                    amethyst_position,
                )
            )
            amethyst_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.AMETHYSTS,
                    state.order_depths[Product.AMETHYSTS],
                    self.params[Product.AMETHYSTS]["fair_value"],
                    self.params[Product.AMETHYSTS]["clear_width"],
                    amethyst_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            amethyst_make_orders, _, _ = self.make_amethyst_orders(
                state.order_depths[Product.AMETHYSTS],
                self.params[Product.AMETHYSTS]["fair_value"],
                amethyst_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.AMETHYSTS]["volume_limit"],
            )
            result[Product.AMETHYSTS] = (
                amethyst_take_orders + amethyst_clear_orders + amethyst_make_orders
            )

        if Product.STARFRUIT in self.params and Product.STARFRUIT in state.order_depths:
            starfruit_position = (
                state.position[Product.STARFRUIT]
                if Product.STARFRUIT in state.position
                else 0
            )
            starfruit_fair_value = self.starfruit_fair_value(
                state.order_depths[Product.STARFRUIT], traderObject
            )
            starfruit_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.STARFRUIT,
                    state.order_depths[Product.STARFRUIT],
                    starfruit_fair_value,
                    self.params[Product.STARFRUIT]["take_width"],
                    starfruit_position,
                    self.params[Product.STARFRUIT]["prevent_adverse"],
                    self.params[Product.STARFRUIT]["adverse_volume"],
                )
            )
            starfruit_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.STARFRUIT,
                    state.order_depths[Product.STARFRUIT],
                    starfruit_fair_value,
                    self.params[Product.STARFRUIT]["clear_width"],
                    starfruit_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            starfruit_make_orders, _, _ = self.make_starfruit_orders(
                state.order_depths[Product.STARFRUIT],
                starfruit_fair_value,
                self.params[Product.STARFRUIT]["starfruit_min_edge"],
                starfruit_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.STARFRUIT] = (
                starfruit_take_orders + starfruit_clear_orders + starfruit_make_orders
            )

        if Product.VOLCANIC_ROCK in self.params and Product.VOLCANIC_ROCK in state.order_depths:
            volcanic_rock_position = (
                state.position[Product.VOLCANIC_ROCK]
                if Product.VOLCANIC_ROCK in state.position
                else 0
            )
            volcanic_rock_fair_value = self.volcanic_rock_fair_value(
                state.order_depths[Product.VOLCANIC_ROCK], traderObject
            )
            volcanic_rock_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.VOLCANIC_ROCK,
                    state.order_depths[Product.VOLCANIC_ROCK],
                    volcanic_rock_fair_value,
                    self.params[Product.VOLCANIC_ROCK]["take_width"],
                    volcanic_rock_position,
                    self.params[Product.VOLCANIC_ROCK]["prevent_adverse"],
                    self.params[Product.VOLCANIC_ROCK]["adverse_volume"],
                )
            )
            volcanic_rock_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.VOLCANIC_ROCK,
                    state.order_depths[Product.VOLCANIC_ROCK],
                    volcanic_rock_fair_value,
                    self.params[Product.VOLCANIC_ROCK]["clear_width"],
                    volcanic_rock_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            volcanic_rock_make_orders, _, _ = self.make_volcanic_rock_orders(
                state.order_depths[Product.VOLCANIC_ROCK],
                volcanic_rock_fair_value,
                self.params[Product.VOLCANIC_ROCK]["starfruit_min_edge"],
                volcanic_rock_position,
                buy_order_volume,
                sell_order_volume,
            )
            result[Product.VOLCANIC_ROCK] = (
                volcanic_rock_take_orders + volcanic_rock_clear_orders + volcanic_rock_make_orders
            )


        # if Product.SQUIDINK in self.params and Product.SQUIDINK in state.order_depths:
        #     squidink_position = (
        #         state.position[Product.SQUIDINK]
        #         if Product.SQUIDINK in state.position
        #         else 0
        #     )
        #     squidink_fair_value = self.squidink_fair_value(
        #         state.order_depths[Product.SQUIDINK], traderObject
        #     )
    
        #     # Get current market prices
        #     best_ask = min(state.order_depths[Product.SQUIDINK].sell_orders.keys())
        #     best_bid = max(state.order_depths[Product.SQUIDINK].buy_orders.keys())
        #     current_spread = best_ask - best_bid
    
        #     squidink_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.SQUIDINK,
        #             state.order_depths[Product.SQUIDINK],
        #             squidink_fair_value,
        #             self.params[Product.SQUIDINK]["take_width"],
        #             squidink_position,
        #             self.params[Product.SQUIDINK]["prevent_adverse"],
        #             self.params[Product.SQUIDINK]["adverse_volume"],
        #         )
        #     )
    
        #     # Aggressive trend following based on MA crossover
        #     signal = traderObject.get("squidink_signal", "HOLD")
        #     if signal == "BUY" and current_spread < 5:
        #         additional_buy = Order(Product.SQUIDINK, best_bid + 1, 
        #                              min(15, self.LIMIT[Product.SQUIDINK] - squidink_position))
        #         squidink_take_orders.append(additional_buy)
    
        #     elif signal == "SELL" and current_spread < 5:
        #         additional_sell = Order(Product.SQUIDINK, best_ask - 1, 
        #                               -min(15, self.LIMIT[Product.SQUIDINK] + squidink_position))
        #         squidink_take_orders.append(additional_sell)
    
        #     squidink_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.SQUIDINK,
        #             state.order_depths[Product.SQUIDINK],
        #             squidink_fair_value,
        #             self.params[Product.SQUIDINK]["clear_width"],
        #             squidink_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
    
        #     squidink_make_orders, _, _ = self.make_orders(
        #         Product.SQUIDINK,
        #         state.order_depths[Product.SQUIDINK],
        #         squidink_fair_value,
        #         squidink_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.SQUIDINK]["disregard_edge"],
        #         self.params[Product.SQUIDINK]["join_edge"],
        #         self.params[Product.SQUIDINK]["default_edge"],
        #     )
    
        #     result[Product.SQUIDINK] = (
        #         squidink_take_orders + squidink_clear_orders + squidink_make_orders
        #     )
        
        if Product.SPREAD1 not in traderObject:
            traderObject[Product.SPREAD1] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position1 = (
            state.position[Product.GIFT_BASKET_1]
            if Product.GIFT_BASKET_1 in state.position
            else 0
        )
        if Product.SPREAD2 not in traderObject:
            traderObject[Product.SPREAD2] = {
                "spread_history": [],
                "prev_zscore": 0,
                "clear_flag": False,
                "curr_avg": 0,
            }

        basket_position2 = (
            state.position[Product.GIFT_BASKET_2]
            if Product.GIFT_BASKET_2 in state.position
            else 0
        )
        spread_orders1 = self.spread_orders1(
            state.order_depths,
            Product.GIFT_BASKET_1,
            basket_position1,
            traderObject[Product.SPREAD1],
        )
        if spread_orders1 != None:
            result[Product.CROISSANTS] = spread_orders1[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders1[Product.JAMS]
            result[Product.DJEMBE] = spread_orders1[Product.DJEMBE]
            result[Product.GIFT_BASKET_1] = spread_orders1[Product.GIFT_BASKET_1]

        spread_orders2 = self.spread_orders2(
            state.order_depths,
            Product.GIFT_BASKET_2,
            basket_position2,
            traderObject[Product.SPREAD2],
        )
        if spread_orders2 != None:
            result[Product.CROISSANTS] = spread_orders2[Product.CROISSANTS]
            result[Product.JAMS] = spread_orders2[Product.JAMS]
            result[Product.GIFT_BASKET_2] = spread_orders2[Product.GIFT_BASKET_2]
        

        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
        
        # if Product.SQUIDINK in self.params and Product.SQUIDINK in state.order_depths:
        #     squidink_position = (
        #         state.position[Product.SQUIDINK]
        #         if Product.SQUIDINK in state.position
        #         else 0
        #     )
        #     squidink_fair_value = self.squidink_fair_value(
        #         state.order_depths[Product.SQUIDINK], traderObject
        #     )
        #     squidink_take_orders, buy_order_volume, sell_order_volume = (
        #         self.take_orders(
        #             Product.SQUIDINK,
        #             state.order_depths[Product.SQUIDINK],
        #             squidink_fair_value,
        #             self.params[Product.SQUIDINK]["take_width"],
        #             squidink_position,
        #             self.params[Product.SQUIDINK]["prevent_adverse"],
        #             self.params[Product.SQUIDINK]["adverse_volume"],
        #         )
        #     )
        #     squidink_clear_orders, buy_order_volume, sell_order_volume = (
        #         self.clear_orders(
        #             Product.SQUIDINK,
        #             state.order_depths[Product.SQUIDINK],
        #             squidink_fair_value,
        #             self.params[Product.SQUIDINK]["clear_width"],
        #             squidink_position,
        #             buy_order_volume,
        #             sell_order_volume,
        #         )
        #     )
        #     squidink_make_orders, _, _ = self.make_orders(
        #         Product.SQUIDINK,
        #         state.order_depths[Product.SQUIDINK],
        #         squidink_fair_value,
        #         squidink_position,
        #         buy_order_volume,
        #         sell_order_volume,
        #         self.params[Product.SQUIDINK]["disregard_edge"],
        #         self.params[Product.SQUIDINK]["join_edge"],
        #         self.params[Product.SQUIDINK]["default_edge"],
        #     )
        #     result[Product.SQUIDINK] = (
        #         squidink_take_orders + squidink_clear_orders + squidink_make_orders
        #     )