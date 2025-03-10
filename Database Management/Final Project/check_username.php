<?php

require_once 'db.php';

@session_start();
function check_has_username($username)
{
	
	
  $result = null;

  
  $sql = "SELECT * FROM `user` WHERE `username` = '{$username}';";

  $query = mysqli_query($_SESSION['link'], $sql);

  //如果請求成功
  if ($query)
  {
    
    if (mysqli_num_rows($query) >= 1)
    {
      
      $result = true;
    }

    mysqli_free_result($query);
  }
  else
  {
    echo "{$sql} 語法執行失敗，錯誤訊息：" . mysqli_error($_SESSION['link']);
  }

  //回傳結果
  return $result;
}

$check = check_has_username($_POST['un']);

if($check)
{
	//若為true 代表有使用者以重複
	echo 'no';
}
else
{
	//若為 null 或者 false 代表沒有使用者，可以註冊
	echo 'yes';	
}

?>