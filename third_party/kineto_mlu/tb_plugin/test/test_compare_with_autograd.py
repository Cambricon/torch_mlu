import copy
import os
import time
import unittest
import pytest
import torch
import torch_mlu
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision
import torchvision.transforms as T
import torchvision.models as models
import torch_tb_profiler.io as io
from torch_tb_profiler.profiler import RunLoader


def create_log_dir():
    log_dir_name = './log{}'.format(str(int(time.time()*1000)))
    try:
        os.makedirs(log_dir_name)
    except Exception:
        raise RuntimeError("Can't create directory: " + log_dir_name)
    return log_dir_name


def get_autograd_result(p, worker_name, record_shapes=False, with_stack=False):
    avgs = p.key_averages()
    sort_by = 'self_mlu_time_total'
    avgs = sorted(
        avgs, key=lambda evt: getattr(evt, sort_by), reverse=True
    )
    is_mlu = False
    if avgs[0].self_mlu_time_total > 0:
        is_mlu = True
    others_prefix = {'enumerate(DataLoader)#', 'Optimizer.zero_grad#', 'Optimizer.step#',
                     'ProfilerStep*',
                     'Memcpy', 'Memset',
                     'cnnl', 'cnrt', 'cnInvokeKernel', 'cnMem'}
    postfix_to_type = {'CPU': 'operator', 'PrivateUse1': 'kernel'}

    def get_type(evt):
        s = str(evt.device_type)
        postfix = s[s.index('.') + 1:]
        evt_type = postfix_to_type[postfix]
        for prefix in others_prefix:
            if evt.key.startswith(prefix):
                evt_type = 'Other'
                break
        return evt_type

    result_dict = dict()
    result_dict[worker_name + '#operator'] = list()
    if is_mlu:
        result_dict[worker_name + '#kernel'] = list()
    for avg in avgs:
        evt_type = get_type(avg)
        if evt_type == 'operator':
            line = [avg.key, int(avg.count)]
            if is_mlu:
                line.extend([int(avg.self_mlu_time_total), int(avg.mlu_time_total)])
            line.extend([int(avg.self_cpu_time_total), int(avg.cpu_time_total)])
            result_dict[worker_name + '#operator'].append(line)
        elif is_mlu and evt_type == 'kernel':
            line = [avg.key, int(avg.count), int(avg.self_mlu_time_total)]
            result_dict[worker_name + '#kernel'].append(line)
    if record_shapes:
        result_dict[worker_name + '#operator#input_shape'] = list()
        avgs = p.key_averages(True)
        sort_by = 'self_mlu_time_total'
        avgs = sorted(
            avgs, key=lambda evt: getattr(evt, sort_by), reverse=True
        )
        for avg in avgs:
            evt_type = get_type(avg)
            if evt_type == 'operator':
                line = [avg.key, str(avg.input_shapes) if avg.input_shapes else '[]', int(avg.count)]
                if is_mlu:
                    line.extend([int(avg.self_mlu_time_total), int(avg.mlu_time_total)])
                line.extend([int(avg.self_cpu_time_total), int(avg.cpu_time_total)])
                result_dict[worker_name + '#operator#input_shape'].append(line)
    # The call stack for legacy and kineto profiler is different for now,
    # The legacy profiler has stack for backward while kineto not
    # So, just disable call stack compare for the moment
    if False and with_stack:
        result_dict[worker_name + '#operator#stack'] = list()
        avgs = p.key_averages(False, 100)
        sort_by = 'self_mlu_time_total'
        avgs = sorted(
            avgs, key=lambda evt: getattr(evt, sort_by), reverse=True
        )
        for avg in avgs:
            evt_type = get_type(avg)
            if evt_type == 'operator' and avg.stack:
                line = [avg.key, int(avg.count)]
                if is_mlu:
                    line.extend([int(avg.self_mlu_time_total), int(avg.mlu_time_total)])
                line.extend([int(avg.self_cpu_time_total), int(avg.cpu_time_total), ''.join(avg.stack)])
                result_dict[worker_name + '#operator#stack'].append(line)

        result_dict[worker_name + '#operator#stack#input_shape'] = list()
        avgs = p.key_averages(True, 100)
        sort_by = 'self_mlu_time_total'
        avgs = sorted(
            avgs, key=lambda evt: getattr(evt, sort_by), reverse=True
        )
        for avg in avgs:
            evt_type = get_type(avg)
            if evt_type == 'operator' and avg.stack:
                line = [avg.key, str(avg.input_shapes), int(avg.count)]
                if is_mlu:
                    line.extend([int(avg.self_mlu_time_total), int(avg.mlu_time_total)])
                line.extend([int(avg.self_cpu_time_total), int(avg.cpu_time_total), ''.join(avg.stack)])
                result_dict[worker_name + '#operator#stack#input_shape'].append(line)

    return result_dict


def generate_plugin_result_row(data):
    row = list()
    row.append(data['name'])
    if 'input_shape' in data:
        row.append(data['input_shape'])
    row.append(data['calls'])
    if 'device_self_duration' in data:
        row.append(data['device_self_duration'])
        row.append(data['device_total_duration'])
    row.extend([data['host_self_duration'], data['host_total_duration']])
    if False and 'call_stack' in data:
        row.append(data['call_stack'])
    return row


def get_plugin_result(run, record_shapes=False, with_stack=False):
    result_dict = dict()
    for (worker_name, span), profile in run.profiles.items():
        worker_name = worker_name.split('.')[0]
        assert profile.operation_table_by_name is not None
        result_dict[worker_name + '#operator'] = list()
        for data in profile.operation_table_by_name['data']:
            row = generate_plugin_result_row(data)
            result_dict[worker_name + '#operator'].append(row)
        if profile.kernel_table is not None:
            rows = profile.kernel_table['data']['rows']
            result_dict[worker_name + '#kernel'] = list()
            for row in rows:
                result_dict[worker_name + '#kernel'].append([row[0], row[2], row[3]])  # row[1] is 'Tensor Cores Used'.
        if record_shapes:
            assert profile.operation_table_by_name_input is not None
            result_dict[worker_name + '#operator#input_shape'] = list()
            for data in profile.operation_table_by_name_input['data']:
                row = generate_plugin_result_row(data)
                result_dict[worker_name + '#operator#input_shape'].append(row)
        # The call stack for legacy and kineto profiler is different for now,
        # The legacy profiler has stack for backward while kineto not
        # So, just disable call stack compare for the moment
        if False and with_stack:
            assert profile.operation_stack_by_name is not None
            assert profile.operation_stack_by_name_input is not None
            result_dict[worker_name + '#operator#stack'] = list()
            op_stack_dict = profile.operation_stack_by_name
            for k, datalist in op_stack_dict.items():
                for data in datalist:
                    row = generate_plugin_result_row(data)
                    result_dict[worker_name + '#operator#stack'].append(row)
            if record_shapes:
                result_dict[worker_name + '#operator#stack#input_shape'] = list()
                op_stack_dict = profile.operation_stack_by_name_input
                for k, datalist in op_stack_dict.items():
                    for data in datalist:
                        row = generate_plugin_result_row(data)
                        result_dict[worker_name + '#operator#stack#input_shape'].append(row)

    return result_dict


def get_train_func(use_mlu=True):
    # Using randome initialization to avoid downloading pretrained weights
    model = models.resnet50(weights=None)
    if use_mlu:
        model.mlu()

    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
    # CIFAR10 dataset is hard to download, using FakeData instead of it
    #trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                        download=True, transform=transform)
    trainset = torchvision.datasets.FakeData(size=10000, image_size=(3, 32, 32), 
                                             num_classes=10, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=2,
                                              shuffle=True, num_workers=0)

    if use_mlu:
        criterion = nn.CrossEntropyLoss().mlu()
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    if use_mlu:
        device = torch.device('mlu:0')
    else:
        device = torch.device('cpu')
    model.train()

    def train(train_step, prof=None):
        for step, data in enumerate(trainloader, 0):
            print('step:{}'.format(step))
            inputs, labels = data[0].to(device=device), data[1].to(device=device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if prof is not None:
                prof.step()
            if step >= train_step:
                break
    return train


def get_output_fn(dir_name, profilers_dict):
    def output_fn(p):
        # In current torch.profiler.profile, at beginning of each span, a new p.profiler will be created.
        # So the same p.profiler will not be shared among different spans
        worker_name = 'worker{}'.format(p.step_num)
        profilers_dict[worker_name] = p.profiler
        tb_trace_handler = torch.profiler.tensorboard_trace_handler(dir_name, worker_name)
        tb_trace_handler(p)
    return output_fn


class TestCompareWithAutogradResult(unittest.TestCase):

    def compare_results(self, log_dir, profilers_dict, use_mlu=True, record_shapes=False, with_stack=False):
        def rm_self_time(row):
            row.pop(-2)
            return row.copy()

        cache = io.Cache()
        loader = RunLoader(os.path.split(log_dir)[-1], log_dir, cache)
        run = loader.load()
        plugin_result = get_plugin_result(run, record_shapes, with_stack)
        count = 0
        for worker_name, p in profilers_dict.items():
            autograd_result = get_autograd_result(p, worker_name, record_shapes, with_stack)
            for key in autograd_result.keys():
                count += 1
                self.assertTrue(key in plugin_result.keys())
                self.assertEqual(len(plugin_result[key]), len(autograd_result[key]))
                # TODO(nwz): for mlu plugin, some cpu ops' self_host_time may be incorrect now.
                autograd_result_copy = copy.deepcopy(autograd_result[key])
                plugin_result_copy = copy.deepcopy(plugin_result[key])
                if key.endswith('#operator') or key.endswith('#operator#input_shape'):
                    autograd_result_copy = list(map(rm_self_time, autograd_result_copy))
                    plugin_result_copy = list(map(rm_self_time, plugin_result_copy))
                for line in plugin_result_copy:
                    self.assertTrue(line in autograd_result_copy)
        self.assertEqual(count, len(plugin_result.keys()))

    #@pytest.mark.skipif(not torch.mlu.is_available(), reason='')
    # @pytest.mark.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    @unittest.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    def test_autograd_api(self):
        with torch.autograd.profiler.profile(use_mlu=True, use_kineto=True, record_shapes=True) as p:
            get_train_func()(5)
        log_dir = create_log_dir()
        p.export_chrome_trace(os.path.join(log_dir, 'worker0.{}.pt.trace.json'.format(int(time.time() * 1000))))
        self.compare_results(log_dir, {'worker0': p})

    def base_profiler_api(self, use_mlu, record_shapes, profile_memory, with_stack):
        log_dir = create_log_dir()
        profilers_dict = dict()
        if use_mlu:
            activities = [
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU]
        else:
            activities = [torch.profiler.ProfilerActivity.CPU]

        with torch.profiler.profile(
            activities=activities,
            schedule=torch.profiler.schedule(
                wait=2,
                warmup=2,
                active=3),
            on_trace_ready=get_output_fn(log_dir, profilers_dict),
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack
        ) as p:
            get_train_func(use_mlu)(13, p)
        self.compare_results(log_dir, profilers_dict, use_mlu, record_shapes, with_stack)
    
    # @pytest.mark.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    @unittest.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    def test_profiler_api_without_mlu(self):
        self.base_profiler_api(False, True, True, False)

    #@pytest.mark.skipif(not torch.mlu.is_available(), reason='')
    # @pytest.mark.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    @unittest.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    def test_profiler_api_with_record_shapes_memory_stack(self):
        self.base_profiler_api(True, True, True, True)

    #@pytest.mark.skipif(not torch.mlu.is_available(), reason='')
    # @pytest.mark.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    @unittest.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    def test_profiler_api_without_record_shapes_memory_stack(self):
        self.base_profiler_api(True, False, False, False)

    #@pytest.mark.skipif(not torch.mlu.is_available(), reason='')
    # @pytest.mark.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    @unittest.skip(reason='There are some issues caused by nsTous and shapes of list tensors')
    def test_profiler_api_without_step(self):
        log_dir = create_log_dir()
        profilers_dict = dict()
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.MLU],
            on_trace_ready=get_output_fn(log_dir, profilers_dict),
            record_shapes=True
        ):
            get_train_func()(7)
        self.compare_results(log_dir, profilers_dict)


if __name__ == '__main__':
    unittest.main()
