import codecs
import json
import os
import pickle
import random
import sys
import time
import uuid
import numpy as np
from flask import *
# https://flask-socketio.readthedocs.io/en/latest/
from flask_socketio import *
from flask_socketio import SocketIO
import logging

logging.getLogger('engineio').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel(logging.ERROR)
timestr = time.strftime('%m%d%H%M')
logger = logging.getLogger("aggregation")

fh = logging.FileHandler('experiments/logs/{}.log'.format(timestr))
# fh = logging.StreamHandler()
fh.setLevel(logging.INFO)
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
logger.addHandler(ch)

from model_wrapper import Models


def load_json(filename):
    with open(filename) as f:
        return json.load(f)


class Aggregator(object):
    """docstring for GlobalModel"""

    def __init__(self, task_config):
        self.task_config = task_config
        self.model_name = task_config['model_name']
        self.current_weights = self.get_init_parameters()
        self.model_path = task_config['model_path']
        # weights should be a ordered list of parameter

        logger.info(self.get_model_description())

        # for stats
        self.train_losses = []
        self.avg_test_losses = []
        self.avg_test_maps = []
        self.avg_test_recalls = []

        # for convergence check
        self.prev_test_loss = None
        self.best_loss = None
        self.best_weight = None
        self.best_round = -1
        self.best_map = 0
        self.best_recall = 0

        self.training_start_time = int(round(time.time()))

    def get_init_parameters(self):
        model = getattr(Models, self.model_name)
        parameters = model(self.task_config).get_weights()
        logger.info("parameters loaded ... delete the model")
        del model
        return parameters

    # client_updates = [(w, n)..]
    def update_weights(self, client_weights, client_sizes):
        total_size = np.sum(client_sizes)
        new_weights = [np.zeros(param.shape) for param in client_weights[0]]
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                new_weights[i] += (client_weights[c][i] * client_sizes[c]
                                   / total_size)
        self.current_weights = new_weights

    def aggregate_loss_map_recall(self, client_losses, client_maps, client_recalls, client_sizes):
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        aggr_maps = sum(client_maps[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        aggr_recalls = sum(client_recalls[i] / total_size * client_sizes[i]
                           for i in range(len(client_sizes)))
        return aggr_loss, aggr_maps, aggr_recalls

    # cur_round could None
    def aggregate_train_loss_accuracy_recall(self, client_losses, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        total_size = sum(client_sizes)
        # weighted sum
        aggr_loss = sum(client_losses[i] / total_size * client_sizes[i]
                        for i in range(len(client_sizes)))
        self.train_losses += [[cur_round, cur_time, aggr_loss]]
        print(aggr_loss)
        return aggr_loss

    # cur_round coule be None
    def aggregate_loss_accuracy_recall(self, client_losses, client_maps, client_recalls, client_sizes, cur_round):
        cur_time = int(round(time.time())) - self.training_start_time
        aggr_loss, aggr_map, aggr_recall = self.aggregate_loss_map_recall(client_losses, client_maps, client_recalls,
                                                                          client_sizes)

        self.avg_test_losses += [[cur_round, cur_time, aggr_loss]]
        self.avg_test_maps += [[cur_round, cur_time, aggr_map]]
        self.avg_test_recalls += [[cur_round, cur_time, aggr_recall]]
        return aggr_loss, aggr_map, aggr_recall

    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "valid_loss": self.valid_losses,
            "train_accuracy": self.train_accuracies,
            "valid_accuracy": self.valid_accuracies,
            "train_recall": self.train_recalls,
            "valid_recall": self.valid_recalls
        }

    def get_model_description(self):
        return "Good morning, Sir."


# Federated Averaging algorithm with the server pulling from clients

class FLServer(object):
    def __init__(self, task_config_filename, host, port):
        self.task_config = load_json(task_config_filename)
        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app, ping_timeout=3600000,
                                 ping_interval=3600000,
                                 max_http_buffer_size=int(1e32))
        self.host = host
        self.port = port
        self.client_resource = {}

        self.MIN_NUM_WORKERS = self.task_config["MIN_NUM_WORKERS"]
        self.MAX_NUM_ROUNDS = self.task_config["MAX_NUM_ROUNDS"]
        self.NUM_TOLERATE = self.task_config["NUM_TOLERATE"]
        self.NUM_CLIENTS_CONTACTED_PER_ROUND = self.task_config["NUM_CLIENTS_CONTACTED_PER_ROUND"]
        self.ROUNDS_BETWEEN_VALIDATIONS = self.task_config["ROUNDS_BETWEEN_VALIDATIONS"]

        self.STOP = False

        self.wait_time = 0
        logger.info(self.task_config)
        self.model_id = str(uuid.uuid4())

        self.aggregator = Aggregator(self.task_config)

        #####
        # training states
        self.current_round = -1  # -1 for not yet started
        self.current_round_client_updates = []
        self.eval_client_updates = []
        #####

        # socket io messages
        self.register_handles()
        self.invalid_tolerate = 0

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.aggregator.get_stats())

    def register_handles(self):
        # single-threaded async, no need to lock

        @self.socketio.on('connect')
        def handle_connect():
            print(request.sid, "connected")

        @self.socketio.on('reconnect')
        def handle_reconnect():
            print(request.sid, "reconnected")

        @self.socketio.on('disconnect')
        def handle_disconnect():
            print(request.sid, "disconnected")
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)

        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            print("client wake_up: ", request.sid)
            emit('init')

        @self.socketio.on('client_ready')
        def handle_client_ready():
            print("client ready for training", request.sid)
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= self.MIN_NUM_WORKERS and self.current_round == -1:
                print("start to federated learning.....")
                self.check_client_resource()
            elif len(self.ready_client_sids) < self.MIN_NUM_WORKERS:
                print("not enough client worker running.....")
            else:
                print("current_round is not equal to -1, please restart server.")

        @self.socketio.on('check_client_resource_done')
        def handle_check_client_resource_done(data):
            if data['round_number'] == self.current_round:
                self.client_resource[request.sid] = data['load_rate']
                if len(self.client_resource) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:
                    satisfy = 0
                    client_sids_selected = []
                    for client_id, val in self.client_resource.items():
                        print(client_id, "cpu rate: ", val)
                        if float(val) < 0.4:
                            client_sids_selected.append(client_id)
                            print(client_id, "satisfy")
                            satisfy = satisfy + 1
                        else:
                            print(client_id, "reject")

                    if satisfy / len(self.client_resource) > 0.5:
                        self.wait_time = min(self.wait_time, 3)
                        time.sleep(self.wait_time)
                        self.train_next_round(client_sids_selected)
                    else:
                        if self.wait_time < 10:
                            self.wait_time = self.wait_time + 1
                        time.sleep(self.wait_time)
                        self.check_client_resource()

        @self.socketio.on('client_update')
        def handle_client_update(data):
            logger.info("received client update of bytes: {}".format(sys.getsizeof(data)))
            logger.info("handle client_update {}".format(request.sid))

            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = pickle_string_to_obj(data['weights'])
                if len(self.current_round_client_updates) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:

                    # current train
                    self.aggregator.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates]
                    )
                    print(self.current_round)
                    print([x['train_loss'] for x in self.current_round_client_updates])
                    print([x['train_size'] for x in self.current_round_client_updates])
                    aggr_train_loss = self.aggregator.aggregate_train_loss_accuracy_recall(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )
                    logger.info("=== training ===")
                    logger.info("aggr_train_loss {}".format(aggr_train_loss))

                    if 'client_test_loss' in self.current_round_client_updates[0]:
                        aggr_test_loss, aggr_test_map, aggr_test_recall = self.aggregator.aggregate_loss_accuracy_recall(
                            [x['client_test_loss'] for x in self.current_round_client_updates],
                            [x['client_test_map'] for x in self.current_round_client_updates],
                            [x['client_test_recall'] for x in self.current_round_client_updates],
                            [x['client_test_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )
                        logger.info("=== aggregation ===")
                        logger.info("aggr_test_loss {}".format(aggr_test_loss))
                        logger.info("aggr_test_map {}".format(aggr_test_map))
                        logger.info("aggr_test_recall {}".format(aggr_test_recall))

#                        if self.aggregator.best_loss is None or (self.aggregator.best_loss > aggr_test_loss):
#                            self.aggregator.best_loss = aggr_test_loss
#                            self.aggregator.best_weight = self.aggregator.current_weights
#                            self.aggregator.best_round = self.current_round
#                            print('current:' + str(self.current_round))
#                            print('best:' + str(self.aggregator.best_round))

                        if self.aggregator.prev_test_loss is not None and self.aggregator.prev_test_loss < aggr_test_loss:
                            self.invalid_tolerate = self.invalid_tolerate + 1
                        else:
                            self.invalid_tolerate = 0

                        self.aggregator.prev_test_loss = aggr_test_loss

                        if self.NUM_TOLERATE > 0 and self.invalid_tolerate > self.NUM_TOLERATE:
                            logger.info("converges! starting test phase..")
                            self.STOP = True

                    if self.current_round >= self.MAX_NUM_ROUNDS:
                        logger.info("get to maximum step, stop...")
                        self.STOP = True

                    self.stop_and_eval()

        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            logger.info("handle client_eval {}".format(request.sid))
            # logger.info("eval_resp {}".format(data))
            self.eval_client_updates += [data]

            # tolerate 30% unresponsive clients
            if len(self.eval_client_updates) == self.NUM_CLIENTS_CONTACTED_PER_ROUND:

                server_test_loss = self.eval_client_updates[0]['test_loss']
                server_test_map = self.eval_client_updates[0]['test_map']
                server_test_recall = self.eval_client_updates[0]['test_recall']
                logger.info("=== server test ===")
                logger.info("server_test_loss {}".format(server_test_loss))
                logger.info("server_test_map {}".format(server_test_map))
                logger.info("server_test_recall {}".format(server_test_recall))

                if self.aggregator.best_map <= server_test_map:
                    self.aggregator.best_map = server_test_map
                    self.aggregator.best_loss = server_test_loss
                    self.aggregator.best_recall = server_test_recall
                    self.aggregator.best_round = self.current_round
                if self.STOP:
                    logger.info("== done ==")
                    self.eval_client_updates = None  # special value, forbid evaling again
                    logger.info("Federated training finished ... ")
                    logger.info("best model at round {}".format(self.aggregator.best_round))
                    logger.info("get best test loss {}".format(self.aggregator.best_loss))
                    logger.info("get best map {}".format(self.aggregator.best_map))
                    logger.info("get best recall {}".format(self.aggregator.best_recall))
                else:
                    logger.info("start to next round...")
                    self.check_client_resource()

    def check_client_resource(self):

        self.client_resource = {}
        client_sids_selected = random.sample(list(self.ready_client_sids), self.NUM_CLIENTS_CONTACTED_PER_ROUND)
        print('send weights')
        for rid in client_sids_selected:
            emit('check_client_resource', {
                'round_number': self.current_round,
            }, room=rid)

    # Note: we assume that during training the #workers will be >= MIN_NUM_WORKERS
    def train_next_round(self, client_sids_selected):
        self.current_round += 1
        # buffers all client updates
        self.current_round_client_updates = []

        logger.info("### Round {} ###".format(self.current_round))

        logger.info("request updates from {}".format(client_sids_selected))
        # by default each client cnn is in its own "room"
        current_weights = obj_to_pickle_string(self.aggregator.current_weights, self.aggregator.model_path)
        for rid in client_sids_selected:
            if self.current_round == 0:
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                    'current_weights': current_weights,
                    'model_path': self.aggregator.model_path,
               #     'aggregation': self.current_round % self.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)
                logger.info("sent initial model to client")
            else:
                emit('request_update', {
                    'model_id': self.model_id,
                    'round_number': self.current_round,
                #    'aggregation': self.current_round % self.ROUNDS_BETWEEN_VALIDATIONS == 0,
                }, room=rid)

    def stop_and_eval(self):
        current_weights = obj_to_pickle_string(self.aggregator.current_weights, self.aggregator.model_path)
        self.eval_client_updates = []
        start = time.time()
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                'model_id': self.model_id,
                'current_weights': current_weights,
                'weights_format': 'pickle',
                'STOP': self.STOP
            }, room=rid)
        logger.info("sent aggregated model to client")

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)


def obj_to_pickle_string(x, file_path=None):
    if file_path is not None:
        print("save model to file")
        output = open(file_path, 'wb')
        pickle.dump(x, output)
        return file_path
    else:
        print("turn model to byte")
        #x = pickle.dumps(x)
        x=codecs.encode(pickle.dumps(x), "base64").decode()
        print(len(x))
        return x
    # return msgpack.packb(x, default=msgpack_numpy.encode)
    # TODO: compare pickle vs msgpack vs json for serialization; tradeoff: computation vs network IO


def pickle_string_to_obj(s):
    if ".pkl" in s:
        df = open(s, "rb")
        print("load model from file")
        return pickle.load(df)
    else:
        print("load model from byte")
        return pickle.loads(codecs.decode(s.encode(), "base64"))
        #return pickle.loads(s)


if __name__ == '__main__':
    # print(sys.argv)

    try:
        task_config_filename = sys.argv[1]
        if not os.path.exists(task_config_filename):
            print(task_config_filename, "doesn't exists")
            raise FileNotFoundError()
    except:
        print("please input the task_config_filename")
        exit(0)
    try:
        server = FLServer(task_config_filename, "127.0.0.1", 5000)
        # server = FLServer(RESNET_MODEL, "127.0.0.1", 5000)
        print("listening on 127.0.0.1:5000")
        server.start()
    except ConnectionError:
        print('Restart server fail.')
