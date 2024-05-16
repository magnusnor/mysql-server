#include "sql/conn_handler/ml_socket_connection.h"

#include "sql/join_optimizer/access_path.h"
#include "sql/join_optimizer/make_join_hypergraph.h"
#include "sql/join_optimizer/node_map.h"
#include "sql/join_optimizer/relational_expression.h"
#include "sql/join_optimizer/print_utils.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <stdio.h>
#include <string>
#include <thread>
#include <regex>

/**
  The MLSocket class represents an abstraction for creating a
  UNIX socket for connecting to a ML model.
*/
MLSocket::MLSocket(const std::string &socket_path)
    : m_socket_path(socket_path), m_socket_fd(-1), m_is_connected(false) {}

MLSocket::~MLSocket() { Disconnect(); }

bool MLSocket::ConnectSocket() {
  int max_retries = 10;
  int retry_interval_seconds = 1;
  if (m_is_connected) {
    printf("[MySQL] Already connected to ML model (%s).\n", m_socket_path.c_str());
    return true;
  }

  m_socket_fd = socket(AF_UNIX, SOCK_STREAM, 0);
  if (m_socket_fd < 0) {
    printf("[MySQL] Failed to create ML UNIX socket\n");
    return true;
  }

  struct sockaddr_un address;
  memset(&address, 0, sizeof(address));
  address.sun_family = AF_UNIX;
  strncpy(address.sun_path, m_socket_path.c_str(),sizeof(address.sun_path) - 1);

  for (int i = 0; i < max_retries; ++i) {
    if (connect(m_socket_fd, (struct sockaddr *)&address, sizeof(address)) == 0) {
      printf("[MySQL] Connected to %s\n", m_socket_path.c_str());
      m_is_connected = true;
      return false;
    }

    printf("[MySQL] Connecting to ML model (%s) failed, retrying...\n", m_socket_path.c_str());
    std::this_thread::sleep_for(std::chrono::seconds(retry_interval_seconds));
  }

  printf("[MySQL] Connecting to ML model (%s) failed after %d retries\n", m_socket_path.c_str(), max_retries);
  close(m_socket_fd);
  return true;
  return false;
}

void MLSocket::Disconnect() {
  close(m_socket_fd);
  m_is_connected = false;
}

bool MLSocket::WriteML(const std::string &data) {
   std::string message = data + "\n";
   ssize_t bytes = send(m_socket_fd, message.c_str(), message.size(), 0);
   if (bytes == -1) {
     printf("[MySQL] Writing to ML model failed\n");
     return true;
   }
  return false;
}

bool MLSocket::ReadML() {
   memset(m_data, 0, m_data_size);
   ssize_t bytes = read(m_socket_fd, m_data, m_data_size - 1);

   if (bytes < 0) {
     printf("[MySQL] Reading from ML model failed\n");
     return true;
   }
   else if (bytes == 0) {
     printf("[MySQL] No data received from ML model\n");
   } else {
     printf("[MySQL] Data received from ML model: %s\n", m_data);
   }
   return false;
 }

 char* MLSocket::GetData() {
   return m_data;
 }

 /**
   The MLQueryRep class represents an abstraction for creating a
   query representation compatible with the ML model.
 */
 MLQueryRep::MLQueryRep() {
   m_operators = {"=", "<", ">", "<=", ">="};
 }
 MLQueryRep::~MLQueryRep() {}

 void MLQueryRep::AddTables(JoinHypergraph *graph, const JoinPredicate *edge) {
   m_tables = "[";
   bool first = true;
   for (size_t node_idx : BitsSetIn(GetNodeMapFromTableMap(edge->expr->tables_in_subtree, graph->table_num_to_node_num))) {
     if (!first) {
       m_tables += ", ";
     }
     first = false;
     m_tables += "\"";
     m_tables += graph->nodes[node_idx].table->s->table_name.str;
     m_tables += " ";
     m_tables += graph->nodes[node_idx].table->alias;
     m_tables += "\"";
   }
   m_tables += "]";
 }

 void MLQueryRep::AddJoins(JoinHypergraph *graph, const JoinPredicate *edge, hypergraph::NodeMap right) {
   std::string join_condition = PrepareCondition(GenerateExpressionLabel(edge->expr).c_str());
   std::string join_condition_left;
   std::string join_condition_right;
   std::string ordered_join_condition = "";

   size_t delimiter_pos = join_condition.find("=");
   if (delimiter_pos != std::string::npos) {
     join_condition_left = join_condition.substr(0, delimiter_pos);
     join_condition_right = join_condition.substr(delimiter_pos + 1);
   }

   for (size_t node_idx : BitsSetIn(right)) {
     std::string node_table = graph->nodes[node_idx].table->alias;
     // If we find the right side table node in the left join condition expression,
     // the join condition is already ordered.
     if (join_condition_left.find(node_table) != std::string::npos) {
       ordered_join_condition = join_condition;
       break;
     }
     // Else, if we find the right side table node in the right join condition expression,
     // move it to the left join condition expression.
     else if (join_condition_right.find(node_table) != std::string::npos && ordered_join_condition.empty()) {
       ordered_join_condition += join_condition_right;
       ordered_join_condition += "=";
       ordered_join_condition += join_condition_left;
     }
     else {
       continue;
     }
   }
   m_joins += "[\"";
   m_joins += ordered_join_condition;
   m_joins += "\"]";
 }

 void MLQueryRep::AddPredicates(JoinHypergraph *graph, const JoinPredicate *edge, AccessPath *left_path, AccessPath *right_path) {
  std::string join_condition = GenerateExpressionLabel(edge->expr).c_str();
  m_predicates += "[";
  ParsePredicates(graph, join_condition, left_path);
  ParsePredicates(graph, join_condition, right_path);
  // If neither left_path nor right_path had any filter_predicates,
  // ensure ML model can parse the predicates array anyway.
  if (m_predicates == "[") {
    m_predicates += "\"\"";
  }
  m_predicates += "]";
 }

 std::string MLQueryRep::PrepareCondition(const std::string &condition) {
   std::string prepared_condition = std::regex_replace(condition, std::regex("[()]"), "");
   prepared_condition.erase(std::remove_if(prepared_condition.begin(), prepared_condition.end(), isspace), prepared_condition.end());
   return prepared_condition;
 }

 std::string MLQueryRep::GetCondition(Item* condition) {
   std::string cond = ItemToString(condition);
   for (const auto& op : m_operators) {
     if (cond.find(op) != std::string::npos) {
       return PrepareCondition(cond);
     }
   }
   return "";
 }

 void MLQueryRep::ParsePredicates(JoinHypergraph *graph, const std::string &join_condition, AccessPath *path) {
  bool first = true;
  for (int pred_idx : BitsSetIn(path->filter_predicates)) {
    Item* condition = graph->predicates[pred_idx].condition;
    if (ItemToString(condition) == join_condition) continue;


    // If there has already been predicates added, and we are iterating the first predicate of path,
    // e.g., already added from AccessPath left_path when we are iterating AccessPath right_path,
    // make sure it continues as comma separated.
    if (m_predicates != "[" && first) {
      m_predicates += ", ";
    }

    if (!first) {
      m_predicates += ", ";
    }

    std::string predicate_condition = GetCondition(condition);
    if (!predicate_condition.empty()) {
      m_predicates += "\"" + predicate_condition + "\"";
    }
    else {
      m_predicates += "\"\"";
    }
    first = false;
  }
 }

 std::string MLQueryRep::CreateQueryRep(JoinHypergraph *graph, const JoinPredicate *edge, AccessPath *left_path, AccessPath *right_path, hypergraph::NodeMap right) {
     AddTables(graph, edge);
     AddJoins(graph, edge, right);
     AddPredicates(graph, edge, left_path, right_path);

     std::string query_rep = "{";
     query_rep += "\"tables\": ";
     query_rep += m_tables;
     query_rep += ",";
     query_rep += "\"joins\": ";
     query_rep += m_joins;
     query_rep += ",";
     query_rep += "\"predicates\": ";
     query_rep += m_predicates;
     query_rep += "}";

     return query_rep;
 }

 /**
   The Ml_model class represents an abstraction for creating a
   ML model object. This object reads/writes to the ML model through the global Ml_socket.
 */
 MLModel::MLModel() : m_cardinality_estimate(1048576.0) {}
 MLModel::~MLModel() {}

 double MLModel::GetCardinalityEstimate(JoinHypergraph *graph, const JoinPredicate *edge, AccessPath *left_path, AccessPath *right_path, hypergraph::NodeMap right) {
   std::string query_rep = m_ml_query_rep.CreateQueryRep(graph, edge, left_path, right_path, right);
   ml_socket.WriteML(query_rep);
   ml_socket.ReadML();
   m_cardinality_estimate = atof(ml_socket.GetData());
   return m_cardinality_estimate;
 }
