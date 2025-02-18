/**
 * create a cluster with 4 nodes
 *
 * - 1 primary
 * - 3 secondary
 *
 * either PRIMARY or the first SECONDARY can be disabled through the HTTP
 * interface
 */

var common_stmts = require("common_statements");
var gr_memberships = require("gr_memberships");

var gr_node_host = "127.0.0.1";

// all nodes are online
var group_replication_members_online =
    gr_memberships.gr_members(gr_node_host, mysqld.global.gr_nodes);
var cluster_nodes = gr_memberships.cluster_nodes(
    mysqld.global.gr_node_host, mysqld.global.cluster_nodes);

var options = {
  group_replication_members: group_replication_members_online,
  innodb_cluster_instances: cluster_nodes,
  metadata_schema_version: [1, 0, 2],
};

var router_select_metadata =
    common_stmts.get("router_select_metadata", options);
var router_select_group_membership =
    common_stmts.get("router_select_group_membership", options);


// primary is removed, first secondary is the new PRIMARY
var gr_members_removed_primary =
    group_replication_members_online.filter(function(el, ndx) {
      return ndx != 0
    });

var cluster_nodes_removed_primary = cluster_nodes.filter(function(el, ndx) {
  return ndx != 0
});

var options_removed_primary = {
  group_replication_members: gr_members_removed_primary,
  innodb_cluster_instances: cluster_nodes_removed_primary,
  metadata_schema_version: [1, 0, 2],
};

var router_select_metadata_removed_primary =
    common_stmts.get("router_select_metadata", options_removed_primary);
var router_select_group_membership_removed_primary =
    common_stmts.get("router_select_group_membership", options_removed_primary);


// first secondary is removed, PRIMARY stays PRIMARY
var gr_members_removed_secondary =
    group_replication_members_online.filter(function(el, ndx) {
      return ndx != 1
    });

var cluster_nodes_removed_secondary = cluster_nodes.filter(function(el, ndx) {
  return ndx != 1
});

var options_removed_secondary = {
  group_replication_members: gr_members_removed_secondary,
  innodb_cluster_instances: cluster_nodes_removed_secondary,
  metadata_schema_version: [1, 0, 2],
};

var router_select_metadata_removed_secondary =
    common_stmts.get("router_select_metadata", options_removed_secondary);
var router_select_group_membership_removed_secondary = common_stmts.get(
    "router_select_group_membership", options_removed_secondary);


// common queries

// prepare the responses for common statements
var common_responses = common_stmts.prepare_statement_responses(
    [
      "router_set_session_options",
      "router_set_gr_consistency_level",
      "router_start_transaction",
      "router_commit",
      "router_select_schema_version",
      "select_port",
      "router_check_member_state",
      "router_select_members_count",
    ],
    options);

if (mysqld.global.primary_removed === undefined) {
  mysqld.global.primary_removed = false;
}

if (mysqld.global.secondary_removed === undefined) {
  mysqld.global.secondary_removed = false;
}

({
  stmts: function(stmt) {
    if (common_responses.hasOwnProperty(stmt)) {
      return common_responses[stmt];
    } else if (stmt === router_select_metadata.stmt) {
      if (mysqld.global.secondary_removed) {
        return router_select_metadata_removed_secondary;
      } else if (mysqld.global.primary_removed) {
        return router_select_metadata_removed_primary;
      } else {
        return router_select_metadata;
      }
    } else if (stmt === router_select_group_membership.stmt) {
      if (mysqld.global.secondary_removed) {
        return router_select_group_membership_removed_secondary;
      } else if (mysqld.global.primary_removed) {
        return router_select_group_membership_removed_primary;
      } else {
        return router_select_group_membership;
      }
    } else {
      return {
        error: {
          code: 1273,
          sql_state: "HY001",
          message: "Syntax Error at: " + stmt
        }
      };
    }
  }
})
