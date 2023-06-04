def check_ov_devices(anchor=""):
    from openvino.runtime import Core
    core = Core()

    def param_to_string(parameters) -> str:
        """Convert a list / tuple of parameters returned from IE to a string."""
        if isinstance(parameters, (list, tuple)):
            return ', '.join([str(x) for x in parameters])
        else:
            return str(parameters)

    infostr = f'[INFO| {anchor}] Available devices:'
    for i, device in enumerate(core.available_devices):
        infostr += f" ({i}) {device} "
        # print('\tSUPPORTED_PROPERTIES:')
        # for property_key in core.get_property(device, 'SUPPORTED_PROPERTIES'):
        #     if property_key not in ('SUPPORTED_METRICS', 'SUPPORTED_CONFIG_KEYS', 'SUPPORTED_PROPERTIES'):
        #         try:
        #             property_val = core.get_property(device, property_key)
        #         except TypeError:
        #             property_val = 'UNSUPPORTED TYPE'
        #         print(f'\t\t{property_key}: {param_to_string(property_val)}')
    print(infostr)

if __name__ == "__main__" :
    check_ov_devices()

    print("joto")