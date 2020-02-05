import argparse
import copy
from datetime import datetime
import glob
import os
import uuid
import yaml

# TODO:
#	reserved words
#		(unnamed, metadata)
#	readability
#	make interactive editor less ridiculous

# Configuration file to use if a value isn't passed
CONF_F = "/home/tchapman/root/bin/config.yaml"

### Interactive editor parameters
argument_separator = " " # currently relevant only for 'set' 1/28/20
# argument_separator = ":"
pretty_cmds = [
    'cd {path}', 'ls [path]', 'ls -r [path]', 'mk {dict}', 
    'set {path/var}'+argument_separator+'{val}', 'exit', 'top', 
    'options', 'help [cmd]', 'name [name]', 'rm {path/var}'
]
all_cmds = [
    'cd', 'ls', 'ls -r', 'mk', 'set', 'exit', 'top', 'options', 
    'help', 'name', 'rm'
]

cmd_dict = dict(zip(all_cmds, pretty_cmds))

help_txt = {
    "cd": "Move your pointer to the named dictionary",
    "ls": "List all parameters and dictionaries in the current " +
          "or passed dictionary.", 
    "ls -r": "List all parameters and dictionaries in the current " +
             "or passed dictionary recursively. This will loop forever " +
             "if called on a dictionary with a loop; don't do that.",
    "mk": "Create a new dictionary in the current dictionary with the " +
          "given name.",
    "set": "Set the paramter {var} equal to {val} in the configuration.",
    "exit": "Save the current configuration and run whatever script " +
            "initiated this.",
    "top": "Return your pointer to the top dictionary of the " +
           "configuration. Equivalent to 'cd /'",
    "options": "List the valid commands.",
    "help": "Provide a short description of the given command.",
    "name": "If a value for [name] is passed, " +
            "set the name this configuration is to be saved under. " +
            "Otherwise, print the current name.",
    "rm": "Remove {var} from the current or given dictionary"
}

# do string concatenation so I can use variables in the YAML files
# this is automatically invoked by yaml.safe_load
class Join( yaml.YAMLObject ):
    yaml_loader = yaml.SafeLoader
    yaml_tag = "!join"
    
    @classmethod
    def from_yaml(cls, loader, node):
        return "".join([s.value for s in node.value])

### Utility methods

# combine the dictionaries d1 and d2 with
# values from d2 replacing those in d1 when keys overlap
# modifies d1 in place
# loops forever if dictionary has a loop
def recursive_merge( d1, d2 ):
    for key, val in d2.items():
        if isinstance( val, dict ):
            d1.setdefault( key, {} )
            # if not merged( d1[key] ):
            recursive_merge( d1[key], val )
        else:
            d1[key] = val
    return d1

# remove all entries in d1 which exist in d2
# loops forever if dictionary has a loop
def diff_config( d1, d2 ):
    out_dict = {}
    for key, val in d1.items():
        if not key in d2:
            out_dict[key] = val
        else:
            if isinstance( val, dict ):
                # if not diffed( d2[key] ):
                tmp_dict = diff_config( val, d2[key] )
                if len( list( tmp_dict.keys() ) ):
                    out_dict[key] = tmp_dict
            elif d1[key] != d2[key]:
                out_dict[key] = val
                
    return out_dict

# load the given configuration file and
# cast all the top level headers to string so I can compare them to strings
def load_config(f):
    config = yaml.safe_load(f)
    out_config = {}
    for key, value in config.items():
        out_config[str(key)] = value
    return out_config

# find and load config
def find_and_load_config( conf_name, main_config_f ):
    return find_and_load_configs( [conf_name], main_config_f )

# conf_name: key of the config to load
# main_config_f: configuration file containing default header
def find_and_load_configs( conf_names, main_config_f ):
    configs = {}
    with open( main_config_f, "r" ) as f:
        main_config = load_config(f)
        default = main_config["default"]
        for conf_name in conf_names:
            if conf_name in main_config:
                configs[conf_name] = main_config[conf_name]

    # if some conf_names weren't in main_config_f...
    folder = os.path.dirname( main_config_f )
    files = glob.glob( os.path.join( folder, "generated_configs_*.yaml" ) )
    for fn in files:
        if len(configs.keys()) == len(conf_names):
            break
        with open( fn, "r" ) as f:
            generated_config = load_config(f)
            for conf_name in conf_names:
                if conf_name in generated_config:
                    configs[conf_name] = generated_config[conf_name]

    if len(configs.keys()) < len(conf_names):
        pass

    for name in conf_names:
        # modifies conf in-place
        recursive_merge( default, configs[name] ) 
    return default

# recursively generate yaml specification of dict config
def dump_config( config, indent=0, recurse=True ):
    text = ""

    keys = list(config.keys())
    keys.sort()
    for k in keys:
        v = config[k]
        if isinstance( v, dict ):
            text += " "*indent + "{}: \n".format(k)
            if recurse:
                # if printed( k ):
                    # text += " "*indent + "[loop]\n"
                # else:
                text += dump_config( v, indent=indent+2 )
        else:
            text += " "*indent + "{}: {}\n".format( k, v )
    return text
    

def save_config( config, config_fn, name="", metadata={} ):
    key = str(uuid.uuid1().int)

    generated = False
    if not name:
        name = key
        generated = True

    # store metadata dictionary with the new configuration
    config["metadata"] = {
        "timestamp": datetime.now(),
	"key": key,
        "name": name
    }
    # add any metadata provided by the invoking script
    recursive_merge( config["metadata"], metadata )

    save_file = config_fn
    if generated:
        folder = os.path.dirname( config_fn )
        year = datetime.now().year
        save_file = os.path.join( folder, "generated_configs_{}.yaml".format( year ) )

    if not os.path.exists( save_file ):
        os.mknod( save_file )

    with open( save_file, "r+" ) as f:
        skipping = False

        d = f.readlines()
        f.seek(0)
        for l in d:
            # erase the previous version of this config if any
            if l.find(name+":") == 0:
                skipping = True
            elif l[0] != " ":
                skipping = False

            if not skipping:
                f.write(l)

        text = dump_config( {name: config} )
        f.write( text )
        f.truncate()
    return name

def build_parser():
    parser = argparse.ArgumentParser(description="Handle config selection and creation")
    parser.add_argument( 'source', nargs="?", default=CONF_F, 
                         help="The configuration file to read from" )
    parser.add_argument( '-i', '--siteid', 
                         help="The site to use the configuration for." )
    parser.add_argument( '-c', '--config', action='append', default=[], 
                         help="Load an existing configuration with key CONFIG. Can be used multiple times; if multiple loaded configs share attributes, the last one given takes precendence." )
    parser.add_argument( '-n', '--new', nargs='?', const="unnamed", 
       help="Create a new configuration file with name NAME (defaults to a random key). Can be used in conjuction with -c to change the initial state of your new configuration." )
    parser.add_argument( '-e', '--edit', default="",
                         help="Edit an existing config file. When used in conjunction with -c, loads the target file and then loads any configs from -c on top of it." )
    parser.add_argument( '-d', '--dates', action='append', default=[], 
                         help="Shortcut to create a new configuration using the given date ranges for pipeline/target_days. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument( '-r', '--reprocess', action='store_true',  
                         help="Shortcut to create a new configuration with pipeline/reprocess=True. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-l', '--leadminutes', action='append', default=[],
                         help="Shortcut to create a new configuration using the given values for pipeline/lead_minutes. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-s', '--sensors', action='append', default=[],
                         help="Shortcut to create a new configuration using the given values for pipeline/sensors. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-u', '--update', nargs=2, action='append', default=[],
                         help="-u Shortcut to create set {var} to {value} under a header given by the invoking script. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    parser.add_argument('-p', '--path', nargs=2, action='append', default=[],
                         help="Shortcut to create {path_name} to {value}. Can be used multiple times. Cannot be used in conjuction with -n or -e" )
    return parser

### Editor methods

# get a valid command from the user and split it into the command
# and a string of arguments
def get_cmd():
    cmd = None 
    while True:
        print( ">>> ", end="" )
        cmd = input().strip().lower()
        n = 0
        for c in all_cmds:
            if len(c) > n:
                n = len(c)
        while n > 0:
            if len(cmd) >= n and cmd[:n] in all_cmds:
                return [cmd[:n], cmd[n:].strip()]
            n -= 1
        print( "Invalid command " + cmd )
        options()

def options():
    print( "Options: [{}]".format(", ".join(pretty_cmds)) )

def show_help( cmd ):
    if not cmd:
        print( "Type 'options' for a list of valid commands" )
        print( "Type 'help [cmd]' for information about cmd" )
    else:
        print( "{}: {}".format(cmd_dict[cmd], help_txt[cmd]) )

def follow_path( current_pointer, path, full_config ):
    target = current_pointer
    # start from the top for absolute paths
    if len(path) and not path[0]:
        target = full_config
        path.pop(0)
    # ignore trailing slashes
    if len(path) and not path[-1]:
        path.pop()

    for d in path:
        if d in target and isinstance(target[d], dict):
            target = target[d]
        else:
            print( "No dictionary {}".format(d) )
            target = current_pointer
            break
    return target

# create a new configuration or edit an existing one
def build_config( full_config, all_configs, config_f, name="", metadata={}, editing=False ):
    if "metadata" in full_config:
        del full_config["metadata"]

    opts = all_cmds

    print( "Welcome to the configuration builder" )
    print( "Type 'help [option]', 'options', or 'exit' at any time" )
    options()
    pointer = full_config
    while True:
        [cmd, params] = get_cmd()
        if params == "":
            pieces = []
        else:
            # limit the number of parameters to 2 because
            # that is currently the most any valid command takes
            # and this way we can easily specify set var [v1, v2, ...]
            pieces = params.split( argument_separator, 1 )

        # special commands
        if cmd == 'exit':
            if name in all_configs and not editing:
                print( "A configuration already exists with the name " + str(name) )
                print( "Overwrite the existing configuration? Y|[n]: " )
                resp = input().strip().lower()
                if resp == "y":
                    break
            else:
                break 
        elif cmd == 'options':
            options()
        elif cmd == 'help':
            help_target = pieces[0] if len(pieces) else ""
            show_help( help_target )

        # editing commands
        if cmd == 'ls' or cmd == 'ls -r':
            recurse = cmd == 'ls -r'
            target = pointer
            if len(pieces):
                path = pieces[0].split( "/" )
                target = follow_path( pointer, path, full_config )

            print( dump_config( target, recurse=recurse) )
        elif cmd == 'set':
            if len(pieces) != 2:
                show_help( 'set' )
            else:
                # TODO if path is invalid, don't set the value
                path = pieces[0].split( "/" )
                target = follow_path( pointer, path[:-1], full_config )
                if not path[-1] in target:
                    print( "Warning: adding new key {}".format(pieces[1]) )
                target[path[-1]] = pieces[1]
        elif cmd == 'top':
            pointer = full_config
        elif cmd == 'cd':
            if len(pieces) != 1:
                show_help('cd')
            else:
                path = pieces[0].split( "/" )
                pointer = follow_path( pointer, path, full_config )
        elif cmd == 'mk':
            if len(pieces) != 1:
                show_help('mk')
            elif not pieces[0] in pointer:
                pointer[pieces[0]] = {}
        elif cmd == "name":
            if not len(pieces) or pieces[0] == "":
                print( "Name is set as: " + name )
            elif editing:
                print( "Name changes are not supported while editing configurations" )
            else:
                name = pieces[0]
        elif cmd == 'rm':
            if not len(pieces):
                show_help( 'rm' )
            else:
                # TODO if path is invalid, don't set the value
                path = pieces[0].split( "/" )
                target = follow_path( pointer, path[:-1], full_config )
                if not path[-1] in target:
                    print( "No such key {} to remove.".format( pieces[0] ) )
                else:
                    del target[path[-1]]
    
    # remove any attributes that match default if save_full_configs is off
    default = {} if all_configs["meta"]["save_full_configs"] else all_configs["default"]
    diff = diff_config( full_config, default )

    name = save_config( diff, config_f, name=name, metadata=metadata )
    return name

### Public facing method
# process given arguments and
# construct and return a configuration accordingly
def handle_config(default_config_name="", metadata={}, parser=None, header=""):
    if header:
        invoking_header = header
    elif metadata and "invoking_script" in metadata:
        invoking_header = metadata["invoking_script"]

    if not parser:
        parser = build_parser()
    args = parser.parse_args()

    config_f = args.source 
    with open( config_f, "r" ) as f:
        all_configs = load_config(f)

    loaded_conf_names = args.config # non-default config files to use

    # shortcuts to run custom pipeline...
    if args.update or args.leadminutes or args.reprocess or args.dates or args.sensors or args.path or args.siteid:
        # process quick-config args and generate an unnamed configuration to use
        lead_minutes = args.leadminutes
        reprocess = args.reprocess
        target_dates = args.dates
        explicit_updates = args.update
        path_updates = args.path
        site_id = args.siteid
    
        # build a new full_config using the loaded files
        conf = find_and_load_configs( loaded_conf_names, config_f )
    
        # update that full_config using the given quick-config arguments
        if lead_minutes:
            conf["pipeline"]["lead_minutes"] = lead_minutes
        if reprocess:
            conf["pipeline"]["reprocess"] = reprocess
        if target_dates:
            conf["pipeline"]["target_dates"] = target_dates
        if site_id:
            conf["site_id"] = site_id
        for p in path_updates:
            conf["paths"][p[0]] = p[1]
        for u in explicit_updates:
            if not invoking_header:
                print( "Error: no header given when config was initialized. Cannot process -u flags." )
                exit(1)
            conf[invoking_header][u[0]] = u[1]
    
        # remove any attributes that match default if save_full_configs is off
        default = {} if all_configs["meta"]["save_full_configs"] else all_configs["default"]
        diff = diff_config( conf, default )
        conf_name = save_config( diff, config_f, name="", metadata=metadata )
        
        # reload the config to deal with some issues with how data
        # entered in the editor is interpreted (e.g arrays are strings)
        return find_and_load_config( conf_name, config_f )
    elif args.new or args.edit:
        if args.edit:
            target_name = args.edit
            if not target_name in all_configs:
                # this doesn't work if someone tries to edit a generated config
                # but they shouldn't be edited; they're there for posterity
                # and run replication
                print( "Cannot edit {}; it is not an existing config. " + 
                       "Try one of {}".format(
                  target_name, ", ".join( [str(s) for s in all_configs.keys()] )
                ) )
                exit(1)

            # build a new full_config using the loaded files
            configs_to_load = [target_name] + loaded_conf_names
            conf = find_and_load_configs( configs_to_load, config_f )
        elif args.new:
            target_name = args.new if args.new != "unnamed" else default_config_name
            # build a new full_config using the loaded files
            conf = find_and_load_configs( loaded_conf_names, config_f )

        conf_name = build_config( conf, all_configs, config_f,
          name=target_name, metadata=metadata, editing=args.edit )
        # reload the config to deal with some issues with how data
        # entered in the editor is interpreted (e.g. arrays are strings)
        return find_and_load_config( conf_name, config_f )
    else:
        # just load the given configurations and return their union
        conf = find_and_load_configs( loaded_conf_names, config_f )

	# for logging purposes:
        if not "metadata" in conf:
            conf["metadata"] = {}
        conf["metadata"]["name"] = loaded_conf_names # for logging
        return conf
