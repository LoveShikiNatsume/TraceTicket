from .queries import Query
from .utils import *
import logging

logger = logging.getLogger("autoquery-scenario")
highspeed_weights = {True: 60, False: 40}


def query_and_cancel(q: Query):
    try:
        if random_from_weighted(highspeed_weights):
            pairs = q.query_orders(types=tuple([0, 1]))
        else:
            pairs = q.query_orders(types=tuple([0, 1]), query_other=True)

        if not pairs:
            print("no orders found for cancel")
            return

        pair = random_from_list(pairs)
        if not pair:
            print("no valid order pair for cancel")
            return

        order_id = q.cancel_order(order_id=pair[0])
        if not order_id:
            return

        logger.info(f"{order_id} queried and canceled")
    except Exception as e:
        print(f"query and cancel error: {e}")


def query_and_collect(q: Query):
    try:
        if random_from_weighted(highspeed_weights):
            pairs = q.query_orders(types=tuple([1]))
        else:
            pairs = q.query_orders(types=tuple([1]), query_other=True)

        if not pairs:
            print("no orders found for collect")
            return

        pair = random_from_list(pairs)
        if not pair:
            print("no valid order pair for collect")
            return

        order_id = q.collect_order(order_id=pair[0])
        if not order_id:
            return

        logger.info(f"{order_id} queried and collected")
    except Exception as e:
        print(f"query and collect error: {e}")


def query_and_execute(q: Query):
    try:
        if random_from_weighted(highspeed_weights):
            pairs = q.query_orders(types=tuple([1]))
        else:
            pairs = q.query_orders(types=tuple([1]), query_other=True)

        if not pairs:
            print("no orders found for execute")
            return

        pair = random_from_list(pairs)
        if not pair:
            print("no valid order pair for execute")
            return

        order_id = q.enter_station(order_id=pair[0])
        if not order_id:
            return

        logger.info(f"{order_id} queried and entered station")
    except Exception as e:
        print(f"query and execute error: {e}")


def query_and_preserve(q: Query):
    try:
        start = ""
        end = ""
        trip_ids = []

        high_speed = random_from_weighted(highspeed_weights)
        if high_speed:
            start = "Shang Hai"
            end = "Su Zhou"
            high_speed_place_pair = (start, end)
            trip_ids = q.query_high_speed_ticket(place_pair=high_speed_place_pair)
        else:
            start = "Shang Hai"
            end = "Nan Jing"
            other_place_pair = (start, end)
            trip_ids = q.query_normal_ticket(place_pair=other_place_pair)

        if not trip_ids:
            print("no trips found for preserve")
            return

        _ = q.query_assurances()
        q.preserve(start, end, trip_ids, high_speed)
    except Exception as e:
        print(f"query and preserve error: {e}")


def query_and_consign(q: Query):
    try:
        if random_from_weighted(highspeed_weights):
            list = q.query_orders_all_info()
        else:
            list = q.query_orders_all_info(query_other=True)

        if not list:
            print("no orders found for consign")
            return

        res = random_from_list(list)
        if not res:
            print("no valid order for consign")
            return

        order_id = q.put_consign(res)
        if not order_id:
            return

        logger.info(f"{order_id} queried and put consign")
    except Exception as e:
        print(f"query and consign error: {e}")


def query_and_pay(q: Query):
    try:
        if random_from_weighted(highspeed_weights):
            pairs = q.query_orders(types=tuple([0, 1]))
        else:
            pairs = q.query_orders(types=tuple([0, 1]), query_other=True)

        if not pairs:
            print("no orders found for pay")
            return

        pair = random_from_list(pairs)
        if not pair:
            print("no valid order pair for pay")
            return

        order_id = q.pay_order(pair[0], pair[1])
        if not order_id:
            return

        logger.info(f"{order_id} queried and paid")
    except Exception as e:
        print(f"query and pay error: {e}")


def query_and_rebook(q: Query):
    try:
        if random_from_weighted(highspeed_weights):
            pairs = q.query_orders(types=tuple([0, 1]))
        else:
            pairs = q.query_orders(types=tuple([0, 1]), query_other=True)

        if not pairs:
            print("no orders found for rebook")
            return

        pair = random_from_list(pairs)
        if not pair:
            print("no valid order pair for rebook")
            return

        order_id = q.cancel_order(order_id=pair[0])
        if not order_id:
            return

        q.rebook_ticket(pair[0], pair[1], pair[1])
        logger.info(f"{order_id} queried and rebooked")
    except Exception as e:
        print(f"query and rebook error: {e}")